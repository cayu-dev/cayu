from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from copy import deepcopy
from decimal import Decimal

import pytest
from tests.core._event_projection_support import (
    private_event_for_public_event,
    private_events_for_public_events,
)
from tests.core._execution_profile_fixtures import (
    rebind_test_invocation,
)

from cayu.core import (
    AgentSpec,
    Event,
    EventType,
    Message,
    ToolResultPart,
)
from cayu.core.tools import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.environments import Environment, EnvironmentSpec
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    BudgetLimit,
    CayuApp,
    EventQuery,
    ExecutionProfileComponentClass,
    ExecutionProfileMismatchError,
    ForkSessionRequest,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    InterruptSessionRequest,
    ResumeRequest,
    RetryPolicy,
    RunLimits,
    RunRequest,
    RuntimeHook,
    Session,
    SessionRuntimePublicationConflict,
    SessionStatus,
    StructuredOutputSpec,
    StructuredOutputStrategy,
    ToolApprovalRecoveryOutcome,
    ToolCallHookContext,
    ToolCapabilityCeiling,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
    ToolRoundIdentity,
    UserInputRecoveryRequest,
    UserInputResponse,
)
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime import sessions as sessions_module
from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY, public_event_sequence
from cayu.runtime.checkpoints import (
    AMBIGUOUS_PENDING_USER_INPUT_CHECKPOINT_KEY,
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    CheckpointCompatibilityError,
    decode_runtime_checkpoint,
)
from cayu.runtime.costs import ModelPrice, PriceBook
from cayu.runtime.execution_profiles import (
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.runtime.user_input import (
    AMBIGUOUS_USER_INPUT_SUPERSESSION_INTENT_KEY,
    AmbiguousUserInputPauseAuthorityError,
    UserInputPauseState,
    user_input_answer_request_digest,
    user_input_resolution_request_digest,
)
from cayu.tools.user_input import UserInputTool
from cayu.vaults import SecretRedactor, StaticVault


class _ScriptedProvider(ModelProvider):
    """First step emits the given tool calls; every later step finishes with text."""

    name = "fake"

    def __init__(self, first_round: list[tuple[str, str, dict]], final_text: str = "done") -> None:
        self._first_round = first_round
        self._final_text = final_text
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            for call_id, name, arguments in self._first_round:
                yield ModelStreamEvent.tool_call(id=call_id, name=name, arguments=arguments)
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta(self._final_text)
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _RunConfigProvider(ModelProvider):
    """Pause for input, request a follow-up tool, then finish."""

    name = "fake"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            yield ModelStreamEvent.tool_call(
                id="call_input",
                name="ask_user",
                arguments={"question": "Continue?"},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        if len(self.requests) == 2:
            yield ModelStreamEvent.tool_call(
                id="call_echo",
                name="echo",
                arguments={"text": "after input"},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _RunCostConfigProvider(_RunConfigProvider):
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        async for event in super().stream(request):
            if event.type == "completed":
                input_tokens = 5 if len(self.requests) == 1 else 6
                yield ModelStreamEvent.completed(
                    {
                        "finish_reason": event.payload.get("finish_reason"),
                        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                    }
                )
            else:
                yield event


class _BlockingContinuationProvider(_ScriptedProvider):
    def __init__(self, first_round: list[tuple[str, str, dict]]) -> None:
        super().__init__(first_round)
        self.continuation_started: asyncio.Event | None = None
        self.never_complete: asyncio.Event | None = None

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            for call_id, name, arguments in self._first_round:
                yield ModelStreamEvent.tool_call(id=call_id, name=name, arguments=arguments)
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        if self.continuation_started is None or self.never_complete is None:
            raise AssertionError("Blocking continuation events were not initialized.")
        self.continuation_started.set()
        await self.never_complete.wait()
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _EchoTool(Tool):
    spec = ToolSpec(
        name="echo",
        description="Echo text.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    def __init__(self) -> None:
        super().__init__()
        self.metadata_by_text: dict[str, dict] = {}

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.metadata_by_text[args["text"]] = ctx.metadata
        return ToolResult(content=args["text"])


class _BlockingTool(Tool):
    spec = ToolSpec(
        name="block",
        description="Block until the consuming runtime task is cancelled.",
        input_schema={"type": "object", "properties": {}},
    )

    def __init__(self) -> None:
        super().__init__()
        self.started: asyncio.Event | None = None
        self.never_complete: asyncio.Event | None = None

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        if self.started is None or self.never_complete is None:
            raise AssertionError("Blocking tool events were not initialized.")
        self.started.set()
        await self.never_complete.wait()
        return ToolResult(content="unexpected")


class _RecordingReleaseStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.release_calls: dict[str, int] = {}

    async def release_session_invocation(self, command) -> None:
        self.release_calls[command.session_id] = self.release_calls.get(command.session_id, 0) + 1
        await super().release_session_invocation(command)


class _CommitThenRaiseReleaseStore(_RecordingReleaseStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.fail_next_release = False

    async def release_session_invocation(self, command) -> None:
        await super().release_session_invocation(command)
        if self.fail_next_release:
            self.fail_next_release = False
            raise RuntimeError("run fence release unavailable")


class _FailingReleaseBeforeCleanupStore(_RecordingReleaseStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.fail_next_release = False

    async def release_session_invocation(self, command) -> None:
        self.release_calls[command.session_id] = self.release_calls.get(command.session_id, 0) + 1
        if self.fail_next_release:
            self.fail_next_release = False
            raise RuntimeError("run fence release unavailable before cleanup")
        await InMemorySessionStore.release_session_invocation(self, command)


class _BlockingCommittedRunningTransitionStore(_RecordingReleaseStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.block_next_running_transition = False
        self.transition_committed: asyncio.Event | None = None
        self.finish_transition: asyncio.Event | None = None
        self.block_next_execution_admission = False
        self.execution_admission_committed: asyncio.Event | None = None
        self.finish_execution_admission: asyncio.Event | None = None
        self.fail_next_execution_admission_acknowledgement = False
        self.execution_admission_acknowledgement_error: BaseException | None = None
        self.execution_admission_process_control: BaseException | None = None
        self.execution_admission_reconciliation_error: BaseException | None = None
        self.block_next_execution_admission_reconciliation = False
        self.execution_admission_reconciliation_started: asyncio.Event | None = None
        self.finish_execution_admission_reconciliation: asyncio.Event | None = None

    async def transition_status_and_checkpoint(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        checkpoint_transform,
        **kwargs,
    ) -> Session:
        session = await super().transition_status_and_checkpoint(
            session_id,
            from_statuses=from_statuses,
            to_status=to_status,
            checkpoint_transform=checkpoint_transform,
            **kwargs,
        )
        if self.block_next_running_transition and to_status == SessionStatus.RUNNING:
            self.block_next_running_transition = False
            if self.transition_committed is None or self.finish_transition is None:
                raise AssertionError("Transition boundary events were not initialized.")
            self.transition_committed.set()
            await self.finish_transition.wait()
        return session

    async def transform_checkpoint(self, session_id, checkpoint_transform) -> None:
        await super().transform_checkpoint(session_id, checkpoint_transform)
        checkpoint = await self.load_checkpoint(session_id)
        intent = None if checkpoint is None else checkpoint.get("user_input_resolution_intent")
        if type(intent) is not dict or intent.get("execution_state") != "executing":
            return
        if self.fail_next_execution_admission_acknowledgement:
            self.fail_next_execution_admission_acknowledgement = False
            acknowledgement_error = self.execution_admission_acknowledgement_error
            self.execution_admission_acknowledgement_error = None
            if acknowledgement_error is not None:
                raise acknowledgement_error
            raise OSError("execution admission acknowledgement lost")
        if self.block_next_execution_admission_reconciliation:
            self.block_next_execution_admission_reconciliation = False
            if (
                self.execution_admission_reconciliation_started is None
                or self.finish_execution_admission_reconciliation is None
            ):
                raise AssertionError("Reconciliation boundary events were not initialized.")
            self.execution_admission_reconciliation_started.set()
            await self.finish_execution_admission_reconciliation.wait()
            reconciliation_error = self.execution_admission_reconciliation_error
            self.execution_admission_reconciliation_error = None
            if reconciliation_error is not None:
                raise reconciliation_error
        elif self.execution_admission_reconciliation_error is not None:
            reconciliation_error = self.execution_admission_reconciliation_error
            self.execution_admission_reconciliation_error = None
            raise reconciliation_error
        if self.execution_admission_process_control is not None:
            process_control = self.execution_admission_process_control
            self.execution_admission_process_control = None
            raise process_control
        if self.block_next_execution_admission:
            self.block_next_execution_admission = False
            if (
                self.execution_admission_committed is None
                or self.finish_execution_admission is None
            ):
                raise AssertionError("Execution admission boundary events were not initialized.")
            self.execution_admission_committed.set()
            await self.finish_execution_admission.wait()


class _BlockingAbandonedFinalizationStore(_RecordingReleaseStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.block_next_interrupted_transition = False
        self.finalization_started: asyncio.Event | None = None
        self.finish_finalization: asyncio.Event | None = None

    async def publish_interaction_transition(
        self,
        session_id: str,
        *,
        event: Event,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        only_if_no_queued_messages: bool = False,
        model_completion_stage_settlement=None,
        expected_session_instance_id: str | None = None,
        expected_active_invocation_profile=None,
        expected_invocation_authority_state="active",
        expected_recovery_claim_id: str | None = None,
        expected_recovery_claim_clock=None,
    ):
        if self.block_next_interrupted_transition and to_status == SessionStatus.INTERRUPTED:
            self.block_next_interrupted_transition = False
            if self.finalization_started is None or self.finish_finalization is None:
                raise AssertionError("Finalization boundary events were not initialized.")
            self.finalization_started.set()
            await self.finish_finalization.wait()
        return await super().publish_interaction_transition(
            session_id,
            event=event,
            from_statuses=from_statuses,
            to_status=to_status,
            only_if_no_queued_messages=only_if_no_queued_messages,
            model_completion_stage_settlement=model_completion_stage_settlement,
            expected_session_instance_id=expected_session_instance_id,
            expected_active_invocation_profile=expected_active_invocation_profile,
            expected_invocation_authority_state=expected_invocation_authority_state,
            expected_recovery_claim_id=expected_recovery_claim_id,
            expected_recovery_claim_clock=expected_recovery_claim_clock,
        )


async def _collect(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]


def _tool_round_identity_payload(events: list[Event]) -> dict[str, str]:
    pause = next(event for event in events if event.type == EventType.SESSION_AWAITING_USER_INPUT)
    return {
        key: pause.payload[key] for key in ("model_step_id", "model_attempt_id", "tool_round_id")
    }


def _crashed_user_input_resume_events(
    events: list[Event],
    *,
    session_id: str,
    tool_call_id: str,
) -> list[Event]:
    """Build the durable prefix emitted before a resumed tool crashes."""

    pause = next(event for event in events if event.type == EventType.SESSION_AWAITING_USER_INPUT)
    input_id = pause.payload["input_id"]
    tool_calls = pause.payload["tool_calls"]
    assert isinstance(input_id, str)
    assert isinstance(tool_calls, list)
    tool_call = next(
        call
        for call in tool_calls
        if isinstance(call, dict) and call.get("tool_call_id") == tool_call_id
    )
    tool_name = tool_call["tool_name"]
    assert isinstance(tool_name, str)
    assert tool_call.get("arguments_state") == "quarantined"
    assert "arguments" not in tool_call
    identity = ToolRoundIdentity.model_validate(_tool_round_identity_payload(events))
    idempotency_key = tool_execution.tool_idempotency_key(
        session_id=session_id,
        tool_round_id=identity.tool_round_id,
        tool_call_id=tool_call_id,
        pause_id=input_id,
    )
    return [
        Event(
            type=EventType.SESSION_RESUMED,
            session_id=session_id,
            interaction_id=pause.interaction_id,
            agent_name=pause.agent_name,
            environment_name=pause.environment_name,
            payload={
                **identity.payload(),
                "interruption_type": "user_input_required",
                "input_id": input_id,
                "tool_call_id": pause.payload["tool_call_id"],
                "resolved_by": None,
            },
        ),
        Event(
            type=EventType.TOOL_CALL_STARTED,
            session_id=session_id,
            interaction_id=pause.interaction_id,
            agent_name=pause.agent_name,
            environment_name=pause.environment_name,
            tool_name=tool_name,
            payload={
                **identity.payload(),
                "input_id": input_id,
                "tool_call_id": tool_call_id,
                "idempotency_key": idempotency_key,
                "arguments_state": "quarantined",
            },
        ),
    ]


def _tool_result_parts(transcript) -> list[ToolResultPart]:
    tool_message = next(message for message in transcript if message.role == "tool")
    return [part for part in tool_message.content if isinstance(part, ToolResultPart)]


def _build(
    first_round,
    *,
    tools=None,
    final_text="done",
    store=None,
    secret_redactor: SecretRedactor | None = None,
):
    store = InMemorySessionStore() if store is None else store
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        secret_redactor=secret_redactor,
    )
    app.register_provider(_ScriptedProvider(first_round, final_text=final_text), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=tools if tools is not None else [UserInputTool()],
    )
    return app, store


def test_ask_user_pauses_the_session() -> None:
    app, store = _build(
        [("call_1", "ask_user", {"question": "Which env?", "options": ["dev", "prod"]})]
    )
    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant", session_id="s_pause", messages=[Message.text("user", "go")]
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_INTERRUPTED
    awaiting = next(e for e in events if e.type == EventType.SESSION_AWAITING_USER_INPUT)
    private_awaiting = asyncio.run(private_event_for_public_event(store, awaiting))
    assert awaiting.payload["question"] == "Which env?"
    assert awaiting.payload["options"] == ["dev", "prod"]
    assert awaiting.payload["input_id"]
    assert [call["tool_call_id"] for call in private_awaiting.payload["tool_calls"]] == ["call_1"]
    interrupted = next(e for e in events if e.type == EventType.SESSION_INTERRUPTED)
    assert interrupted.payload["interruption_type"] == "user_input_required"
    assert asyncio.run(store.load("s_pause")).status == SessionStatus.INTERRUPTED
    checkpoint = asyncio.run(store.load_checkpoint("s_pause"))
    assert checkpoint[CHECKPOINT_SCHEMA_VERSION_KEY] == CURRENT_CHECKPOINT_SCHEMA_VERSION
    assert "pending_user_input" in checkpoint


@pytest.mark.parametrize("secret", ["pause_digest", "source_run_epoch"])
def test_ask_user_open_authority_survives_secret_key_collision(secret: str) -> None:
    async def run() -> None:
        session_id = (
            "s_pause_key_collision_digest"
            if secret == "pause_digest"
            else "s_pause_key_collision_epoch"
        )
        app, store = _build(
            [("call_1", "ask_user", {"question": "Continue?"})],
            secret_redactor=SecretRedactor(secret),
        )
        events = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
        checkpoint_event = next(
            event
            for event in events
            if event.type is EventType.SESSION_CHECKPOINTED
            and event.payload.get("checkpoint") == "pending_user_input"
        )
        awaiting_event = next(
            event for event in events if event.type is EventType.SESSION_AWAITING_USER_INPUT
        )
        for event in (checkpoint_event, awaiting_event):
            assert "source_run_epoch" in event.payload
            assert "pause_digest" in event.payload

        private_events = await private_events_for_public_events(
            store,
            [checkpoint_event, awaiting_event],
        )
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        pending = checkpoint["pending_user_input"]
        receipt = await store.load_runtime_publication_receipt(
            session_id,
            f"user-input-open:{pending['input_id']}",
        )
        assert receipt is not None
        assert receipt.intent["pause_digest"] == private_events[0].payload["pause_digest"]
        assert all(
            event.payload["source_run_epoch"] == pending["source_run_epoch"]
            for event in private_events
        )
        session = await store.load(session_id)
        assert session is not None and session.status is SessionStatus.INTERRUPTED

    asyncio.run(run())


def test_answered_user_input_transition_survives_secret_collisions() -> None:
    async def run() -> None:
        session_id = "s_user_input_close_control_collision"
        app, store = _build(
            [("call_1", "ask_user", {"question": "Continue?"})],
            secret_redactor=SecretRedactor(
                [
                    "transition",
                    "answered",
                    "resolution_request_digest",
                    "execution_state",
                    "claimed",
                    "executing",
                ]
            ),
        )
        paused = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
        awaiting = next(
            event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
        )

        resolved = await _drain(
            app.resolve_user_input(
                UserInputResponse(
                    session_id=session_id,
                    input_id=awaiting.payload["input_id"],
                    answer="yes",
                )
            )
        )
        close = next(
            event
            for event in resolved
            if event.type is EventType.SESSION_CHECKPOINTED
            and event.payload.get("transition") == "answered"
        )
        private_close = await private_event_for_public_event(store, close)

        assert close.payload["transition"] == "answered"
        assert close.payload["pause_digest"] == PRIVATE_EVENT_AUTHORITY
        assert close.payload["resolution_request_digest"] == PRIVATE_EVENT_AUTHORITY
        assert private_close.payload["transition"] == "answered"
        assert len(private_close.payload["pause_digest"]) == 64
        assert len(private_close.payload["resolution_request_digest"]) == 64

    asyncio.run(run())


def test_resolve_user_input_injects_answer_and_continues() -> None:
    app, store = _build(
        [("call_1", "ask_user", {"question": "Which env?"})],
        final_text="Deploying to prod.",
    )
    app.register_environment(
        Environment(EnvironmentSpec(name="optional")),
        default=False,
    )
    pause_events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                environment_name="optional",
                session_id="s_resume",
                messages=[Message.text("user", "go")],
            ),
        )
    )
    awaiting = next(e for e in pause_events if e.type == EventType.SESSION_AWAITING_USER_INPUT)
    input_id = awaiting.payload["input_id"]
    private_input_id = asyncio.run(private_event_for_public_event(store, awaiting)).payload[
        "input_id"
    ]
    app.register_environment(
        Environment(EnvironmentSpec(name="later-default")),
        default=True,
    )

    resume_events = asyncio.run(
        _drain(
            app.resolve_user_input(
                UserInputResponse(session_id="s_resume", input_id=input_id, answer="prod")
            )
        )
    )

    assert resume_events[-1].type == EventType.SESSION_COMPLETED
    assert {event.environment_name for event in [*pause_events, *resume_events]} == {"optional"}
    session = asyncio.run(store.load("s_resume"))
    assert session is not None
    assert session.environment_name == "optional"
    private_resume_events = asyncio.run(private_events_for_public_events(store, resume_events))
    started = next(
        event
        for event in private_resume_events
        if event.type == EventType.TOOL_CALL_STARTED
        and event.payload.get("tool_call_id") == "call_1"
    )
    assert started.payload["effect"] == ToolEffect.EXTERNAL.value
    completed = next(
        event
        for event in private_resume_events
        if event.type == EventType.TOOL_CALL_COMPLETED
        and event.payload.get("tool_call_id") == "call_1"
    )
    assert completed.payload["idempotency_key"] == tool_execution.tool_idempotency_key(
        session_id="s_resume",
        tool_round_id=completed.payload["tool_round_id"],
        tool_call_id="call_1",
        pause_id=private_input_id,
    )
    assert asyncio.run(store.load("s_resume")).status == SessionStatus.COMPLETED
    parts = _tool_result_parts(asyncio.run(store.load_transcript("s_resume")))
    ask_part = next(part for part in parts if part.tool_call_id == "call_1")
    assert ask_part.content == "prod"
    assert ask_part.is_error is False
    assert "pending_user_input" not in asyncio.run(store.load_checkpoint("s_resume"))


def test_resolve_user_input_rejects_versionless_root_with_reserved_authority() -> None:
    async def run() -> tuple[SessionStatus, dict]:
        session_id = "s_resume_versionless_root_checkpoint"
        app, store = _build(
            [("call_1", "ask_user", {"question": "Which env?"})],
            final_text="Deploying.",
        )
        pause_events = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
        input_id = next(
            event for event in pause_events if event.type == EventType.SESSION_AWAITING_USER_INPUT
        ).payload["input_id"]
        versionless = await store.load_checkpoint(session_id)
        assert versionless is not None
        versionless.pop(CHECKPOINT_SCHEMA_VERSION_KEY, None)
        versionless["future_additive_field"] = {"kept": True}
        with sessions_module._invocation_lifecycle_authority_mutation_scope():
            await store.checkpoint(session_id, versionless)

        with pytest.raises(AmbiguousUserInputPauseAuthorityError):
            await _drain(
                app.resolve_user_input(
                    UserInputResponse(
                        session_id=session_id,
                        input_id=input_id,
                        answer="prod",
                    )
                )
            )
        session = await store.load(session_id)
        checkpoint = await store.load_checkpoint(session_id)
        assert session is not None
        assert checkpoint is not None
        return session.status, checkpoint

    status, checkpoint = asyncio.run(run())

    assert status is SessionStatus.INTERRUPTED
    assert CHECKPOINT_SCHEMA_VERSION_KEY not in checkpoint
    assert checkpoint["future_additive_field"] == {"kept": True}
    assert "pending_user_input" in checkpoint
    assert active_invocation_execution_profile_from_checkpoint(checkpoint) is not None


def test_future_root_checkpoint_blocks_user_input_resume_before_governed_work() -> None:
    async def run() -> tuple[CheckpointCompatibilityError, int, SessionStatus, dict]:
        session_id = "s_future_root_checkpoint"
        store = InMemorySessionStore()
        provider = _ScriptedProvider(
            [("call_1", "ask_user", {"question": "Which env?"})],
            final_text="must not run",
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool()],
        )
        pause_events = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
        input_id = next(
            event for event in pause_events if event.type == EventType.SESSION_AWAITING_USER_INPUT
        ).payload["input_id"]
        future_checkpoint = await store.load_checkpoint(session_id)
        assert future_checkpoint is not None
        future_checkpoint[CHECKPOINT_SCHEMA_VERSION_KEY] = CURRENT_CHECKPOINT_SCHEMA_VERSION + 1
        with sessions_module._invocation_lifecycle_authority_mutation_scope():
            await store.checkpoint(session_id, future_checkpoint)

        with pytest.raises(CheckpointCompatibilityError) as caught:
            await _drain(
                app.resolve_user_input(
                    UserInputResponse(
                        session_id=session_id,
                        input_id=input_id,
                        answer="prod",
                    )
                )
            )
        session = await store.load(session_id)
        checkpoint_after = await store.load_checkpoint(session_id)
        assert session is not None
        assert checkpoint_after is not None
        return caught.value, len(provider.requests), session.status, checkpoint_after

    error, provider_calls, status, checkpoint_after = asyncio.run(run())

    assert error.reason == "checkpoint_schema_version_too_new"
    assert provider_calls == 1
    assert status is SessionStatus.INTERRUPTED
    assert checkpoint_after[CHECKPOINT_SCHEMA_VERSION_KEY] == (
        CURRENT_CHECKPOINT_SCHEMA_VERSION + 1
    )
    assert "pending_user_input" in checkpoint_after


def test_resolve_user_input_releases_run_fence_once_after_handoff() -> None:
    async def resolve(*, close_after_handoff: bool) -> tuple[int, SessionStatus, bool]:
        session_id = "s_release_close" if close_after_handoff else "s_release_success"
        store = _RecordingReleaseStore()
        app, _ = _build(
            [("call_1", "ask_user", {"question": "Which env?"})],
            store=store,
        )
        pause_events = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
        awaiting = next(
            event for event in pause_events if event.type == EventType.SESSION_AWAITING_USER_INPUT
        )
        input_id = awaiting.payload["input_id"]

        releases_before_resolution = store.release_calls[session_id]
        stream = app.resolve_user_input(
            UserInputResponse(session_id=session_id, input_id=input_id, answer="prod")
        )
        if close_after_handoff:
            while (await anext(stream)).type != EventType.MODEL_STARTED:
                pass
            await stream.aclose()
        else:
            await _drain(stream)
        session = await store.load(session_id)
        assert session is not None
        return (
            store.release_calls[session_id] - releases_before_resolution,
            session.status,
            app._session_control.has_active_tasks(session_id),
        )

    success_releases, success_status, success_has_active_tasks = asyncio.run(
        resolve(close_after_handoff=False)
    )
    close_releases, close_status, close_has_active_tasks = asyncio.run(
        resolve(close_after_handoff=True)
    )

    assert (success_releases, success_status) == (1, SessionStatus.COMPLETED)
    assert (close_releases, close_status) == (1, SessionStatus.INTERRUPTED)
    assert success_has_active_tasks is False
    assert close_has_active_tasks is False


def test_resolve_user_input_task_cancellation_reconciles_release_acknowledgement_loss() -> None:
    async def run() -> None:
        session_id = "s_resolution_task_cancelled"
        store = _CommitThenRaiseReleaseStore()
        blocking_tool = _BlockingTool()
        app, _ = _build(
            [
                ("call_input", "ask_user", {"question": "Continue?"}),
                ("call_block", "block", {}),
            ],
            tools=[UserInputTool(), blocking_tool],
            store=store,
        )
        blocking_tool.started = asyncio.Event()
        blocking_tool.never_complete = asyncio.Event()
        pause_events = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
        awaiting = next(
            event for event in pause_events if event.type == EventType.SESSION_AWAITING_USER_INPUT
        )
        input_id = awaiting.payload["input_id"]
        private_input_id = (await private_event_for_public_event(store, awaiting)).payload[
            "input_id"
        ]

        releases_before = store.release_calls[session_id]
        store.fail_next_release = True
        resolution_task = asyncio.create_task(
            _drain(
                app.resolve_user_input(
                    UserInputResponse(session_id=session_id, input_id=input_id, answer="yes")
                )
            )
        )
        await asyncio.wait_for(blocking_tool.started.wait(), timeout=5)
        assert resolution_task.cancelling() == 0
        resolution_task.cancel("cancel user-input resolution")
        assert resolution_task.cancelling() == 1
        try:
            await resolution_task
        except asyncio.CancelledError as cancellation:
            assert cancellation.args == ("cancel user-input resolution",)
            assert not any(
                "run fence release" in note for note in getattr(cancellation, "__notes__", ())
            )
        else:
            pytest.fail("User-input resolution did not preserve task cancellation.")

        assert resolution_task.cancelled() is True
        assert resolution_task.cancelling() == 1
        session = await store.load(session_id)
        assert session is not None
        assert session.status == SessionStatus.INTERRUPTED
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert checkpoint["pending_user_input"]["input_id"] == private_input_id
        events = await store.load_events(session_id)
        assert events[-1].type == EventType.SESSION_INTERRUPTED
        assert events[-1].payload["abandoned"] is True
        assert store.release_calls[session_id] - releases_before == 1
        assert app._session_control.has_active_tasks(session_id) is False

    asyncio.run(run())


def test_resolve_user_input_cancellation_after_running_transition_finalizes_claim() -> None:
    async def run() -> None:
        session_id = "s_resolution_cancelled_after_running_commit"
        store = _BlockingCommittedRunningTransitionStore()
        app, _ = _build(
            [("call_input", "ask_user", {"question": "Continue?"})],
            store=store,
        )
        pause_events = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
        awaiting = next(
            event for event in pause_events if event.type == EventType.SESSION_AWAITING_USER_INPUT
        )
        input_id = awaiting.payload["input_id"]
        private_input_id = (await private_event_for_public_event(store, awaiting)).payload[
            "input_id"
        ]

        releases_before = store.release_calls[session_id]
        store.transition_committed = asyncio.Event()
        store.finish_transition = asyncio.Event()
        store.block_next_running_transition = True
        resolution_task = asyncio.create_task(
            _drain(
                app.resolve_user_input(
                    UserInputResponse(session_id=session_id, input_id=input_id, answer="yes")
                )
            )
        )
        await asyncio.wait_for(store.transition_committed.wait(), timeout=5)
        committed = await store.load(session_id)
        assert committed is not None
        assert committed.status == SessionStatus.RUNNING

        resolution_task.cancel("cancel after running transition committed")
        store.finish_transition.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await resolution_task

        assert raised.value.args == ("cancel after running transition committed",)
        assert resolution_task.cancelled() is True
        session = await store.load(session_id)
        assert session is not None
        assert session.status == SessionStatus.INTERRUPTED
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert checkpoint["pending_user_input"]["input_id"] == private_input_id
        assert store.release_calls[session_id] - releases_before == 1
        assert app._session_control.has_active_tasks(session_id) is False

        checkpoint_before_conflict = await store.load_checkpoint(session_id)
        session_before_conflict = await store.load(session_id)
        with pytest.raises(
            SessionRuntimePublicationConflict,
            match="already claimed with a different resolution request",
        ):
            await _drain(
                app.resolve_user_input(
                    UserInputResponse(
                        session_id=session_id,
                        input_id=input_id,
                        answer="yes",
                        max_steps=1,
                    )
                )
            )
        assert await store.load_checkpoint(session_id) == checkpoint_before_conflict
        assert await store.load(session_id) == session_before_conflict

    asyncio.run(run())


def test_resolve_user_input_execution_admission_acknowledgement_loss_reconciles() -> None:
    async def run() -> None:
        session_id = "s_resolution_execution_admission_ack_lost"
        store = _BlockingCommittedRunningTransitionStore()
        app, _ = _build(
            [("call_input", "ask_user", {"question": "Continue?"})],
            store=store,
        )
        paused = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
        input_id = next(
            event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
        ).payload["input_id"]

        store.fail_next_execution_admission_acknowledgement = True
        resolved = await _drain(
            app.resolve_user_input(
                UserInputResponse(session_id=session_id, input_id=input_id, answer="yes")
            )
        )

        assert resolved[-1].type is EventType.SESSION_COMPLETED
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "pending_user_input" not in checkpoint
        assert "user_input_resolution_intent" not in checkpoint

    asyncio.run(run())


def test_resolve_user_input_cancellation_after_execution_admission_is_retryable() -> None:
    async def run() -> None:
        session_id = "s_resolution_cancelled_after_execution_admission"
        store = _BlockingCommittedRunningTransitionStore()
        app, _ = _build(
            [("call_input", "ask_user", {"question": "Continue?"})],
            store=store,
        )
        paused = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
        input_id = next(
            event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
        ).payload["input_id"]
        response = UserInputResponse(session_id=session_id, input_id=input_id, answer="yes")

        store.execution_admission_committed = asyncio.Event()
        store.finish_execution_admission = asyncio.Event()
        store.block_next_execution_admission = True
        resolving = asyncio.create_task(_drain(app.resolve_user_input(response)))
        await asyncio.wait_for(store.execution_admission_committed.wait(), timeout=5)

        resolving.cancel("cancel after execution admission committed")
        store.finish_execution_admission.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await resolving
        assert raised.value.args == ("cancel after execution admission committed",)
        assert resolving.cancelled() is True

        interrupted = await store.load(session_id)
        checkpoint = await store.load_checkpoint(session_id)
        assert interrupted is not None and interrupted.status is SessionStatus.INTERRUPTED
        assert checkpoint is not None
        assert checkpoint["user_input_resolution_intent"]["execution_state"] == "executing"

        retried = await _drain(app.resolve_user_input(response))
        assert retried[-1].type is EventType.SESSION_COMPLETED
        final_checkpoint = await store.load_checkpoint(session_id)
        assert final_checkpoint is not None
        assert "pending_user_input" not in final_checkpoint
        assert "user_input_resolution_intent" not in final_checkpoint

    asyncio.run(run())


@pytest.mark.parametrize(
    "resolution_stage",
    ["answer", "manual-recovery"],
    ids=["answer", "manual-recovery"],
)
@pytest.mark.parametrize(
    "failure_phase",
    ["admission", "reconciliation"],
    ids=["admission", "reconciliation"],
)
def test_user_input_execution_admission_preserves_post_commit_process_control(
    resolution_stage: str,
    failure_phase: str,
) -> None:
    async def run() -> None:
        session_id = f"s_resolution_process_control_{resolution_stage}"
        store = _BlockingCommittedRunningTransitionStore()
        counting = _CountingTool()
        provider = _ScriptedProvider(
            [("call_count", "count", {}), ("call_input", "ask_user", {"question": "q"})],
            final_text="must not dispatch",
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool(), counting],
        )
        paused = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
        awaiting = next(
            event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
        )
        input_id = awaiting.payload["input_id"]
        answer = UserInputResponse(session_id=session_id, input_id=input_id, answer="a")

        if resolution_stage == "manual-recovery":
            await store.append_events(
                session_id,
                _crashed_user_input_resume_events(
                    await private_events_for_public_events(store, paused),
                    session_id=session_id,
                    tool_call_id="call_count",
                ),
            )
            stuck = await _drain(app.resolve_user_input(answer))
            assert stuck[-1].payload.get("manual_recovery_required") is True
            request = UserInputRecoveryRequest(
                session_id=session_id,
                input_id=input_id,
                answer="a",
                tool_call_id="call_count",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="verified completed",
            )
            operation = app.recover_user_input(request)
        else:
            operation = app.resolve_user_input(answer)

        process_control = BaseExceptionGroup(
            f"execution admission {failure_phase} control",
            [SystemExit("shutdown after commit"), RuntimeError("settlement evidence")],
        )
        if failure_phase == "admission":
            store.execution_admission_process_control = process_control
        else:
            store.fail_next_execution_admission_acknowledgement = True
            store.execution_admission_reconciliation_error = process_control
        try:
            await _drain(operation)
        except SystemExit as caught:
            assert caught.args == ("shutdown after commit",)
            assert caught.__cause__ is not None
            assert caught.__cause__.subgroup(RuntimeError) is not None
            if failure_phase == "reconciliation":
                assert caught.__cause__.subgroup(OSError) is not None
        else:
            raise AssertionError("Execution admission suppressed process control.")

        interrupted = await store.load(session_id)
        checkpoint = await store.load_checkpoint(session_id)
        assert interrupted is not None and interrupted.status is SessionStatus.INTERRUPTED
        assert checkpoint is not None
        assert checkpoint["user_input_resolution_intent"]["execution_state"] == "executing"
        assert counting.calls == 0
        assert len(provider.requests) == 1

    asyncio.run(run())


def test_user_input_execution_admission_cancellation_retains_reconciliation_failure() -> None:
    async def run() -> None:
        session_id = "s_resolution_cancelled_during_failed_reconciliation"
        store = _BlockingCommittedRunningTransitionStore()
        counting = _CountingTool()
        app, _ = _build(
            [
                ("call_count", "count", {}),
                ("call_input", "ask_user", {"question": "Continue?"}),
            ],
            tools=[UserInputTool(), counting],
            store=store,
        )
        paused = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
        input_id = next(
            event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
        ).payload["input_id"]

        store.fail_next_execution_admission_acknowledgement = True
        store.execution_admission_reconciliation_error = RuntimeError("reconciliation unavailable")
        store.execution_admission_reconciliation_started = asyncio.Event()
        store.finish_execution_admission_reconciliation = asyncio.Event()
        store.block_next_execution_admission_reconciliation = True
        resolving = asyncio.create_task(
            _drain(
                app.resolve_user_input(
                    UserInputResponse(session_id=session_id, input_id=input_id, answer="yes")
                )
            )
        )
        await asyncio.wait_for(store.execution_admission_reconciliation_started.wait(), timeout=5)

        resolving.cancel("cancel during reconciliation")
        store.finish_execution_admission_reconciliation.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await resolving
        assert raised.value.args == ("cancel during reconciliation",)
        assert resolving.cancelling() == 1
        assert resolving.cancelled() is True
        cause = raised.value.__cause__
        assert isinstance(cause, BaseExceptionGroup)
        assert [type(failure) for failure in cause.exceptions] == [OSError, RuntimeError]
        assert counting.calls == 0

    asyncio.run(run())


def test_user_input_execution_admission_deduplicates_reused_failure_identity() -> None:
    async def run() -> None:
        session_id = "s_resolution_reused_reconciliation_failure"
        store = _BlockingCommittedRunningTransitionStore()
        app, _ = _build(
            [("call_input", "ask_user", {"question": "Continue?"})],
            store=store,
        )
        paused = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
        input_id = next(
            event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
        ).payload["input_id"]

        shared_failure = OSError("shared admission failure")
        store.fail_next_execution_admission_acknowledgement = True
        store.execution_admission_acknowledgement_error = shared_failure
        store.execution_admission_reconciliation_error = shared_failure
        with pytest.raises(OSError) as raised:
            await _drain(
                app.resolve_user_input(
                    UserInputResponse(session_id=session_id, input_id=input_id, answer="yes")
                )
            )
        assert raised.value is shared_failure
        assert raised.value.__cause__ is not raised.value

    asyncio.run(run())


def test_resolve_user_input_repeated_cancellation_cannot_interrupt_finalization() -> None:
    async def run() -> None:
        session_id = "s_resolution_repeated_cancel_during_finalization"
        store = _BlockingAbandonedFinalizationStore()
        blocking_tool = _BlockingTool()
        app, _ = _build(
            [
                ("call_input", "ask_user", {"question": "Continue?"}),
                ("call_block", "block", {}),
            ],
            tools=[UserInputTool(), blocking_tool],
            store=store,
        )
        blocking_tool.started = asyncio.Event()
        blocking_tool.never_complete = asyncio.Event()
        pause_events = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
        awaiting = next(
            event for event in pause_events if event.type == EventType.SESSION_AWAITING_USER_INPUT
        )
        input_id = awaiting.payload["input_id"]
        private_input_id = (await private_event_for_public_event(store, awaiting)).payload[
            "input_id"
        ]

        releases_before = store.release_calls[session_id]
        store.finalization_started = asyncio.Event()
        store.finish_finalization = asyncio.Event()
        store.block_next_interrupted_transition = True
        resolution_task = asyncio.create_task(
            _drain(
                app.resolve_user_input(
                    UserInputResponse(session_id=session_id, input_id=input_id, answer="yes")
                )
            )
        )
        await asyncio.wait_for(blocking_tool.started.wait(), timeout=5)
        resolution_task.cancel("first cancellation")
        await asyncio.wait_for(store.finalization_started.wait(), timeout=5)
        resolution_task.cancel("second cancellation")
        store.finish_finalization.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await resolution_task

        assert raised.value.args == ("first cancellation",)
        assert resolution_task.cancelled() is True
        assert resolution_task.cancelling() == 2
        session = await store.load(session_id)
        assert session is not None
        assert session.status == SessionStatus.INTERRUPTED
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert checkpoint["pending_user_input"]["input_id"] == private_input_id
        assert store.release_calls[session_id] - releases_before == 1
        assert app._session_control.has_active_tasks(session_id) is False

    asyncio.run(run())


def test_resolve_user_input_aclose_surfaces_precleanup_fence_release_failure() -> None:
    async def run() -> None:
        session_id = "s_resolution_aclose_release_failure"
        store = _FailingReleaseBeforeCleanupStore()
        app, _ = _build(
            [("call_input", "ask_user", {"question": "Continue?"})],
            store=store,
        )
        pause_events = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
        input_id = next(
            event for event in pause_events if event.type == EventType.SESSION_AWAITING_USER_INPUT
        ).payload["input_id"]

        releases_before = store.release_calls[session_id]
        stream = app.resolve_user_input(
            UserInputResponse(session_id=session_id, input_id=input_id, answer="yes")
        )
        while (await anext(stream)).type != EventType.MODEL_STARTED:
            pass
        store.fail_next_release = True
        loop = asyncio.get_running_loop()
        reported_contexts: list[dict[str, object]] = []
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: reported_contexts.append(context))
        try:
            with pytest.raises(
                RuntimeError,
                match="run fence release unavailable before cleanup",
            ):
                await stream.aclose()
            await asyncio.sleep(0)
            assert reported_contexts == []
        finally:
            loop.set_exception_handler(previous_handler)

        session = await store.load(session_id)
        assert session is not None
        assert session.status == SessionStatus.INTERRUPTED
        assert store.release_calls[session_id] - releases_before == 1
        assert app._session_control.has_active_tasks(session_id) is False

    asyncio.run(run())


def test_resolve_user_input_events_carry_resolved_by_actor() -> None:
    from cayu import ResolutionActor, ResolutionActorSource

    app, store = _build(
        [("call_1", "ask_user", {"question": "Which env?"})],
        final_text="Deploying to prod.",
    )
    pause_events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_actor",
                messages=[Message.text("user", "go")],
            ),
        )
    )
    input_id = next(
        e for e in pause_events if e.type == EventType.SESSION_AWAITING_USER_INPUT
    ).payload["input_id"]
    app.register_environment(
        Environment(EnvironmentSpec(name="later-default")),
        default=True,
    )

    resume_events = asyncio.run(
        _drain(
            app.resolve_user_input(
                UserInputResponse(
                    session_id="s_actor",
                    input_id=input_id,
                    answer="prod",
                    resolved_by=ResolutionActor(
                        subject="operator@example.com",
                        source=ResolutionActorSource.REQUEST,
                    ),
                )
            )
        )
    )

    assert {event.environment_name for event in [*pause_events, *resume_events]} == {None}
    session = asyncio.run(store.load("s_actor"))
    assert session is not None
    assert session.environment_name is None

    # `claims` stay on the request and are excluded from event payloads.
    expected_actor = {
        "subject": "operator@example.com",
        "tenant": None,
        "source": "request",
    }
    resumed = next(e for e in resume_events if e.type == EventType.SESSION_RESUMED)
    assert resumed.payload["resolved_by"] == expected_actor
    private_resume_events = asyncio.run(private_events_for_public_events(store, resume_events))
    answered = next(
        e
        for e in private_resume_events
        if e.type == EventType.TOOL_CALL_COMPLETED and e.payload.get("tool_call_id") == "call_1"
    )
    assert answered.payload["resolved_by"] == expected_actor
    assert asyncio.run(store.load("s_actor")).status == SessionStatus.COMPLETED


def _run_config_app() -> tuple[CayuApp, InMemorySessionStore, _EchoTool]:
    store = InMemorySessionStore()
    echo = _EchoTool()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(_RunConfigProvider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[UserInputTool(), echo],
    )
    return app, store, echo


@pytest.mark.parametrize("restate_configuration", [False, True])
def test_resolve_user_input_restores_original_run_configuration(
    restate_configuration: bool,
) -> None:
    app, store, echo = _run_config_app()
    session_id = "s_input_restores_run_config"

    pause_events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
                max_steps=7,
                limits=RunLimits(max_tool_calls=1, scope="session"),
                retry_policy=RetryPolicy(max_attempts=3),
            ),
        )
    )
    input_id = next(
        event for event in pause_events if event.type == EventType.SESSION_AWAITING_USER_INPUT
    ).payload["input_id"]
    checkpoint = asyncio.run(store.load_checkpoint(session_id))
    assert checkpoint is not None
    pending = checkpoint["pending_user_input"]
    assert pending["max_steps"] == 7
    assert pending["limits"]["max_tool_calls"] == 1
    assert pending["limits"]["scope"] == "session"
    assert pending["retry_policy"]["max_attempts"] == 3
    assert pending["budget_limits"] == []

    events = asyncio.run(
        _drain(
            app.resolve_user_input(
                UserInputResponse(
                    session_id=session_id,
                    input_id=input_id,
                    answer="yes",
                    max_steps=7 if restate_configuration else None,
                    limits=(
                        RunLimits(max_tool_calls=1, scope="session")
                        if restate_configuration
                        else None
                    ),
                    retry_policy=(RetryPolicy(max_attempts=3) if restate_configuration else None),
                )
            )
        )
    )

    assert echo.metadata_by_text == {}
    limit_events = [event for event in events if event.type == EventType.SESSION_LIMIT_REACHED]
    assert len(limit_events) == 1
    assert limit_events[0].payload["limit"] == "tool_calls"
    session = asyncio.run(store.load(session_id))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


def test_resolve_user_input_rejects_explicit_limits_drift_before_dispatch() -> None:
    app, store, echo = _run_config_app()
    session_id = "s_input_overrides_run_config"

    pause_events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
                limits=RunLimits(max_tool_calls=1, scope="session"),
            ),
        )
    )
    input_id = next(
        event for event in pause_events if event.type == EventType.SESSION_AWAITING_USER_INPUT
    ).payload["input_id"]

    before = asyncio.run(store.load(session_id))
    assert before is not None
    with pytest.raises(ExecutionProfileMismatchError) as caught:
        asyncio.run(
            _drain(
                app.resolve_user_input(
                    UserInputResponse(
                        session_id=session_id,
                        input_id=input_id,
                        answer="yes",
                        limits=RunLimits(),
                    )
                )
            )
        )

    assert caught.value.changed_component_classes == (ExecutionProfileComponentClass.FINALIZATION,)
    assert echo.metadata_by_text == {}
    after = asyncio.run(store.load(session_id))
    assert after is not None
    assert after.status is before.status
    assert after.run_epoch == before.run_epoch


@pytest.mark.parametrize("override_kind", ["limits", "budget_limits"])
def test_resolve_user_input_rejects_limit_or_budget_drift_before_dispatch(
    override_kind: str,
) -> None:
    store = InMemorySessionStore()
    provider = _RunCostConfigProvider()
    echo = _EchoTool()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[UserInputTool(), echo],
    )
    session_id = f"s_input_preserves_{override_kind}"
    cost_limit = BudgetLimit(
        scope="run",
        max_estimated_cost=Decimal("0.000007"),
        pricing=PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="fake",
                    model="fake-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            )
        ),
    )
    pause_events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
                limits=(
                    RunLimits(scope="run")
                    if override_kind == "limits"
                    else RunLimits(max_total_tokens=10, scope="run")
                ),
                budget_limits=(cost_limit,),
            ),
        )
    )
    input_id = next(
        event for event in pause_events if event.type == EventType.SESSION_AWAITING_USER_INPUT
    ).payload["input_id"]

    with pytest.raises(ExecutionProfileMismatchError) as caught:
        asyncio.run(
            _drain(
                app.resolve_user_input(
                    UserInputResponse(
                        session_id=session_id,
                        input_id=input_id,
                        answer="yes",
                        limits=(
                            RunLimits(max_total_tokens=1_000, scope="run")
                            if override_kind == "limits"
                            else None
                        ),
                        budget_limits=() if override_kind == "budget_limits" else None,
                    )
                )
            )
        )

    assert caught.value.changed_component_classes == (
        ExecutionProfileComponentClass.FINALIZATION
        if override_kind == "limits"
        else ExecutionProfileComponentClass.INVOCATION_BUDGET_POLICY,
    )
    assert len(provider.requests) == 1
    assert echo.metadata_by_text == {}


def test_mixed_round_executes_other_tools_and_keeps_model_order() -> None:
    # Model emits [echo, ask_user, echo] in one step. Nothing runs before the pause; on
    # resume the echoes execute and the ask_user answer is injected, all in model order.
    echo = _EchoTool()
    app, store = _build(
        [
            ("call_1", "echo", {"text": "first"}),
            ("call_2", "ask_user", {"question": "continue?"}),
            ("call_3", "echo", {"text": "third"}),
        ],
        tools=[UserInputTool(), echo],
    )
    pause_events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant", session_id="s_mixed", messages=[Message.text("user", "go")]
            ),
        )
    )
    # No echo ran before the pause.
    assert not any(e.type == EventType.TOOL_CALL_COMPLETED for e in pause_events)
    input_id = next(
        e for e in pause_events if e.type == EventType.SESSION_AWAITING_USER_INPUT
    ).payload["input_id"]
    awaiting = next(e for e in pause_events if e.type == EventType.SESSION_AWAITING_USER_INPUT)
    private_awaiting = asyncio.run(private_event_for_public_event(store, awaiting))
    private_input_id = private_awaiting.payload["input_id"]
    active_profile = active_invocation_execution_profile_from_checkpoint(
        asyncio.run(store.load_checkpoint("s_mixed"))
    )
    assert active_profile is not None
    assert [call["tool_call_id"] for call in private_awaiting.payload["tool_calls"]] == [
        "call_1",
        "call_2",
        "call_3",
    ]

    resume_events = asyncio.run(
        _drain(
            app.resolve_user_input(
                UserInputResponse(session_id="s_mixed", input_id=input_id, answer="yes")
            )
        )
    )
    assert resume_events[-1].type == EventType.SESSION_COMPLETED
    private_resume_events = asyncio.run(private_events_for_public_events(store, resume_events))
    attributed_events = [
        event
        for event in resume_events
        if event.type
        in {
            EventType.SESSION_RESUMED,
            EventType.TOOL_CALL_STARTED,
            EventType.TOOL_CALL_COMPLETED,
        }
    ]
    assert attributed_events
    assert {event.payload.get("execution_profile_fingerprint") for event in attributed_events} == {
        active_profile.profile.fingerprint
    }
    sibling_events = [
        event
        for event in private_resume_events
        if event.type in {EventType.TOOL_CALL_STARTED, EventType.TOOL_CALL_COMPLETED}
        and event.payload.get("tool_call_id") in {"call_1", "call_3"}
    ]
    assert sibling_events
    for event in sibling_events:
        call_id = event.payload["tool_call_id"]
        assert event.payload["input_id"] == private_input_id
        assert event.payload["idempotency_key"] == tool_execution.tool_idempotency_key(
            session_id="s_mixed",
            tool_round_id=event.payload["tool_round_id"],
            tool_call_id=call_id,
            pause_id=private_input_id,
        )
    assert echo.metadata_by_text["first"]["input_id"] == private_input_id
    assert echo.metadata_by_text["third"]["input_id"] == private_input_id

    parts = _tool_result_parts(asyncio.run(store.load_transcript("s_mixed")))
    assert [part.tool_call_id for part in parts] == ["call_1", "call_2", "call_3"]
    by_id = {part.tool_call_id: part for part in parts}
    assert by_id["call_1"].content == "first"
    assert by_id["call_2"].content == "yes"
    assert by_id["call_3"].content == "third"


def test_ask_user_is_opt_in_not_registered_by_default() -> None:
    # An agent without UserInputTool registered does not pause; the ask_user call is an
    # ordinary unregistered-tool error and the run proceeds.
    app, _store = _build([("call_1", "ask_user", {"question": "hi"})], tools=[])
    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant", session_id="s_optin", messages=[Message.text("user", "go")]
            ),
        )
    )
    assert not any(e.type == EventType.SESSION_AWAITING_USER_INPUT for e in events)
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_user_input_resume_does_not_execute_tool_registered_after_policy_plan() -> None:
    """Registration drift cannot replace a paused invocation's frozen profile."""

    class FinalProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    echo = _EchoTool()
    first_app, store = _build(
        [
            ("call_late", "echo", {"text": "must not execute"}),
            ("call_input", "ask_user", {"question": "continue?"}),
        ],
        tools=[UserInputTool()],
    )
    pause_events = asyncio.run(
        _collect(
            first_app,
            RunRequest(
                agent_name="assistant",
                session_id="s_registration_drift",
                messages=[Message.text("user", "go")],
            ),
        )
    )
    awaiting = next(
        event for event in pause_events if event.type is EventType.SESSION_AWAITING_USER_INPUT
    )
    pending_calls = awaiting.payload["tool_calls"]
    assert pending_calls[0]["policy_evidence"] == "unregistered"
    assert pending_calls[1]["policy_evidence"] == "authoritative"

    final_provider = FinalProvider()
    resumed_app = CayuApp(session_store=store, enable_logging=False)
    resumed_app.register_provider(final_provider, default=True)
    resumed_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[UserInputTool(), echo],
    )
    with pytest.raises(ExecutionProfileMismatchError) as caught:
        asyncio.run(
            _drain(
                resumed_app.resolve_user_input(
                    UserInputResponse(
                        session_id="s_registration_drift",
                        input_id=awaiting.payload["input_id"],
                        answer="yes",
                    )
                )
            )
        )
    assert caught.value.changed_component_classes == (
        ExecutionProfileComponentClass.DIRECT_TOOLS,
        ExecutionProfileComponentClass.EFFECT_AUTHORITY,
        ExecutionProfileComponentClass.PROVIDER_ADAPTER,
        ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS,
    )
    session = asyncio.run(store.load("s_registration_drift"))
    assert session is not None
    assert session.tool_capability_ceiling == ToolCapabilityCeiling(tool_names=("ask_user",))
    assert echo.metadata_by_text == {}
    assert final_provider.requests == []


def test_ask_user_pauses_whole_round_before_any_tool_runs() -> None:
    # A round mixing ask_user with another (parallel-safe) tool pauses before ANY tool runs,
    # so the sibling never executes until the caller answers. Exercises the pause under main's
    # default-on parallel engine (a multi-call round would otherwise run concurrently).
    app, _store = _build(
        [
            ("call_1", "echo", {"text": "should-not-run"}),
            ("call_2", "ask_user", {"question": "which?"}),
        ],
        tools=[UserInputTool(), _EchoTool()],
    )
    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant", session_id="s_par", messages=[Message.text("user", "go")]
            ),
        )
    )
    assert any(e.type == EventType.SESSION_AWAITING_USER_INPUT for e in events)
    assert events[-1].type == EventType.SESSION_INTERRUPTED
    # Nothing in the round ran before the pause — the echo sibling never started.
    assert not any(e.type == EventType.TOOL_CALL_STARTED for e in events)


def test_stale_user_input_answer_cannot_clear_current_pause() -> None:
    app, store = _build([("call_1", "ask_user", {"question": "q"})])
    asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant", session_id="s_bad", messages=[Message.text("user", "go")]
            ),
        )
    )
    checkpoint_before = asyncio.run(store.load_checkpoint("s_bad"))
    assert checkpoint_before is not None
    current_input_id = checkpoint_before["pending_user_input"]["input_id"]
    with pytest.raises(ValueError, match="does not match pending input"):
        asyncio.run(
            _drain(
                app.resolve_user_input(
                    UserInputResponse(session_id="s_bad", input_id="ui_nope", answer="x")
                )
            )
        )
    assert asyncio.run(store.load_checkpoint("s_bad")) == checkpoint_before
    assert (
        asyncio.run(
            store.load_runtime_publication_receipt(
                "s_bad",
                f"user-input-close:{current_input_id}",
            )
        )
        is None
    )


def test_exact_answer_retry_replays_after_a_newer_pause_opens() -> None:
    class TwoQuestionProvider(ModelProvider):
        name = "two-question"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            request_number = len(self.requests)
            if request_number <= 2:
                yield ModelStreamEvent.tool_call(
                    id=f"call_input_{request_number}",
                    name="ask_user",
                    arguments={"question": f"Question {request_number}?"},
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    async def run() -> None:
        session_id = "s_exact_old_answer_after_new_pause"
        store = InMemorySessionStore()
        provider = TwoQuestionProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool()],
        )

        first_events = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "ask twice")],
            ),
        )
        first_pause = next(
            event for event in first_events if event.type is EventType.SESSION_AWAITING_USER_INPUT
        )
        first_checkpoint = await store.load_checkpoint(session_id)
        assert first_checkpoint is not None
        first_private_input_id = first_checkpoint["pending_user_input"]["input_id"]
        response = UserInputResponse(
            session_id=session_id,
            input_id=first_pause.payload["input_id"],
            answer="first answer",
        )

        second_events = await _drain(app.resolve_user_input(response))
        second_pause = next(
            event for event in second_events if event.type is EventType.SESSION_AWAITING_USER_INPUT
        )
        checkpoint_before_retry = await store.load_checkpoint(session_id)
        assert checkpoint_before_retry is not None
        assert checkpoint_before_retry["pending_user_input"]["input_id"] != first_private_input_id
        assert second_pause.payload["input_id"] != first_pause.payload["input_id"]
        close_receipt = await store.load_runtime_publication_receipt(
            session_id,
            f"user-input-close:{first_private_input_id}",
        )
        assert close_receipt is not None

        retry_events = await _drain(app.resolve_user_input(response))

        assert len(retry_events) == 1
        retry_private_event = await private_event_for_public_event(store, retry_events[0])
        assert retry_private_event.id in close_receipt.appended_event_ids
        assert await store.load_checkpoint(session_id) == checkpoint_before_retry
        assert len(provider.requests) == 2

        with pytest.raises(
            SessionRuntimePublicationConflict,
            match="conflicting resolution authority",
        ):
            await _drain(
                app.resolve_user_input(response.model_copy(update={"answer": "conflicting retry"}))
            )
        assert await store.load_checkpoint(session_id) == checkpoint_before_retry
        assert len(provider.requests) == 2

    asyncio.run(run())


def test_pause_classifier_refreshes_a_stale_snapshot_after_exact_supersession() -> None:
    async def run() -> None:
        session_id = "s_stale_pause_classifier"
        app, store = _build([("call_1", "ask_user", {"question": "Continue?"})])
        await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "ask")],
            ),
        )
        stale_session = await store.load(session_id)
        stale_checkpoint = await store.load_checkpoint(session_id)
        assert stale_session is not None
        assert stale_checkpoint is not None
        private_input_id = stale_checkpoint["pending_user_input"]["input_id"]

        interrupted = await _drain(
            app.interrupt_session(
                InterruptSessionRequest(
                    session_id=session_id,
                    reason="supersede before stale classification",
                )
            )
        )
        assert interrupted[-1].type is EventType.SESSION_INTERRUPTED

        state = await app._recovery_coordinator._classify_user_input_pause(
            session=stale_session,
            checkpoint=stale_checkpoint,
            input_id=private_input_id,
        )

        assert state is UserInputPauseState.SUPERSEDED
        current_checkpoint = await store.load_checkpoint(session_id)
        assert current_checkpoint is not None
        assert "pending_user_input" not in current_checkpoint

    asyncio.run(run())


@pytest.mark.parametrize(
    "secret",
    [
        None,
        AMBIGUOUS_USER_INPUT_SUPERSESSION_INTENT_KEY,
        "ambiguous",
    ],
)
def test_supported_pre_authority_pause_is_reported_ambiguous_and_explicitly_retired(
    secret: str | None,
) -> None:
    async def run() -> None:
        session_id = "s_pre_authority_pause"
        app, store = _build(
            [("call_1", "ask_user", {"question": "Historical question?"})],
            secret_redactor=None if secret is None else SecretRedactor(secret),
        )
        pause_events = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "ask")],
            ),
        )
        public_pause = next(
            event for event in pause_events if event.type is EventType.SESSION_AWAITING_USER_INPUT
        )

        raw_checkpoint = deepcopy(store._checkpoints[session_id])
        historical_pending = raw_checkpoint["pending_user_input"]
        for field_name in (
            "schema_version",
            "session_id",
            "session_instance_id",
            "source_interaction_id",
            "source_run_epoch",
            "execution_profile_fingerprint",
        ):
            historical_pending.pop(field_name, None)
        raw_checkpoint[CHECKPOINT_SCHEMA_VERSION_KEY] = 5
        migrated_checkpoint = decode_runtime_checkpoint(
            raw_checkpoint,
            session_id=session_id,
        )
        assert migrated_checkpoint is not None
        store._checkpoints[session_id] = migrated_checkpoint

        migrated = await store.load_checkpoint(session_id)
        assert migrated is not None
        assert "pending_user_input" not in migrated
        assert AMBIGUOUS_PENDING_USER_INPUT_CHECKPOINT_KEY in migrated
        recovery = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        assert recovery.actions == (IncompleteSessionRecoveryAction.AMBIGUOUS_PENDING_USER_INPUT,)
        assert await store.load_checkpoint(session_id) == migrated

        with pytest.raises(AmbiguousUserInputPauseAuthorityError):
            await _drain(
                app.resolve_user_input(
                    UserInputResponse(
                        session_id=session_id,
                        input_id=public_pause.payload["input_id"],
                        answer="must not execute",
                    )
                )
            )

        interrupt_events = await _drain(
            app.interrupt_session(
                InterruptSessionRequest(
                    session_id=session_id,
                    reason="retire ambiguous historical pause",
                )
            )
        )
        assert interrupt_events[-1].type is EventType.SESSION_INTERRUPTED
        assert interrupt_events[-1].payload["interruption_type"] == "operator_requested"
        assert (
            interrupt_events[-1].payload[AMBIGUOUS_USER_INPUT_SUPERSESSION_INTENT_KEY]["state"]
            == "ambiguous"
        )
        final_checkpoint = await store.load_checkpoint(session_id)
        assert final_checkpoint is not None
        assert AMBIGUOUS_PENDING_USER_INPUT_CHECKPOINT_KEY not in final_checkpoint
        assert "pending_user_input" not in final_checkpoint

    asyncio.run(run())


def test_user_input_close_uses_bounded_round_lifecycle_lookup() -> None:
    class RecordingLifecycleStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.round_lifecycle_lookups = 0

        async def load_tool_round_lifecycle_events_for_round(
            self,
            session_id: str,
            tool_call_ids: list[str] | tuple[str, ...],
            *,
            tool_round_identity: ToolRoundIdentity,
        ) -> list[Event]:
            self.round_lifecycle_lookups += 1
            return await super().load_tool_round_lifecycle_events_for_round(
                session_id,
                tool_call_ids,
                tool_round_identity=tool_round_identity,
            )

    async def run() -> None:
        session_id = "s_bounded_user_input_close"
        store = RecordingLifecycleStore()
        app, _ = _build(
            [("call_1", "ask_user", {"question": "Continue?"})],
            store=store,
        )
        paused = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "ask")],
            ),
        )
        awaiting = next(
            event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
        )

        await _drain(
            app.resolve_user_input(
                UserInputResponse(
                    session_id=session_id,
                    input_id=awaiting.payload["input_id"],
                    answer="yes",
                )
            )
        )

        assert store.round_lifecycle_lookups == 1

    asyncio.run(run())


def test_resolve_user_input_unknown_session_raises() -> None:
    app, _store = _build([("call_1", "ask_user", {"question": "q"})])
    # never run -> session does not exist
    with pytest.raises(KeyError, match="Session not found"):
        asyncio.run(
            _drain(
                app.resolve_user_input(
                    UserInputResponse(session_id="missing", input_id="ui", answer="x")
                )
            )
        )


def test_resolve_user_input_no_pending_raises() -> None:
    # A session that exists but is not awaiting input -> RuntimeError (the "no pending" branch,
    # distinct from the unknown-session KeyError).
    app, _store = _build([("call_1", "echo", {"text": "x"})], tools=[_EchoTool()])
    asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant", session_id="s_np", messages=[Message.text("user", "go")]
            ),
        )
    )
    with pytest.raises(RuntimeError, match="no pending user input"):
        asyncio.run(
            _drain(
                app.resolve_user_input(
                    UserInputResponse(session_id="s_np", input_id="ui", answer="x")
                )
            )
        )


def test_resume_rejects_session_awaiting_user_input() -> None:
    app, _store = _build([("call_1", "ask_user", {"question": "q"})])
    asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant", session_id="s_rej", messages=[Message.text("user", "go")]
            ),
        )
    )
    with pytest.raises(RuntimeError, match="awaiting user input"):
        asyncio.run(
            _drain(
                app.resume(
                    ResumeRequest(session_id="s_rej", messages=[Message.text("user", "more")])
                )
            )
        )


class _DenyEchoPolicy(ToolPolicy):
    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        if request.tool_name == "echo":
            return ToolPolicyResult(decision=ToolPolicyDecision.DENY, reason="echo is denied")
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


class _DenyAskUserPolicy(ToolPolicy):
    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        if request.tool_name == "ask_user":
            return ToolPolicyResult(decision=ToolPolicyDecision.DENY, reason="ask_user is denied")
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


class _DenyFirstAskPolicy(ToolPolicy):
    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        if request.tool_name == "ask_user" and request.arguments.get("question") == "denied-q":
            return ToolPolicyResult(decision=ToolPolicyDecision.DENY, reason="denied")
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


def test_denied_ask_user_does_not_starve_a_later_allowed_one() -> None:
    # A DENY on the first ask_user must not suppress the whole round's pause: a later, allowed
    # ask_user in the same round still pauses (the denied one is blocked on resume).
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(
        _ScriptedProvider(
            [
                ("call_1", "ask_user", {"question": "denied-q"}),
                ("call_2", "ask_user", {"question": "allowed-q"}),
            ]
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[UserInputTool()],
        tool_policy=_DenyFirstAskPolicy(),
    )
    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_starve",
                messages=[Message.text("user", "go")],
            ),
        )
    )
    awaiting = next(e for e in events if e.type == EventType.SESSION_AWAITING_USER_INPUT)
    private_awaiting = asyncio.run(private_event_for_public_event(store, awaiting))
    assert private_awaiting.payload["tool_call_id"] == "call_2"
    assert awaiting.payload["question"] == "allowed-q"


def test_denied_ask_user_does_not_pause() -> None:
    # A tool policy DENY on the ask_user call is enforced by normal execution (blocked), NOT by
    # pausing — otherwise a denied ask_user would still pause and inject the answer as success.
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(
        _ScriptedProvider([("call_1", "ask_user", {"question": "q"})]),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[UserInputTool()],
        tool_policy=_DenyAskUserPolicy(),
    )
    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_denyask",
                messages=[Message.text("user", "go")],
            ),
        )
    )
    assert not any(e.type == EventType.SESSION_AWAITING_USER_INPUT for e in events)
    assert any(e.type == EventType.TOOL_CALL_BLOCKED for e in events)
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert "pending_user_input" not in (asyncio.run(store.load_checkpoint("s_denyask")) or {})


def test_denied_sibling_is_blocked_not_executed_on_resume() -> None:
    # A round [denied echo, ask_user] pauses on ask_user (DENY does not trigger an approval
    # pause). On resume the denied echo must be BLOCKED, not executed (regression: check_policy
    # =False did not re-enforce DENY).
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(
        _ScriptedProvider(
            [
                ("call_1", "echo", {"text": "SHOULD_NOT_RUN"}),
                ("call_2", "ask_user", {"question": "q"}),
            ]
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[UserInputTool(), _EchoTool()],
        tool_policy=_DenyEchoPolicy(),
    )
    pause_events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant", session_id="s_deny", messages=[Message.text("user", "go")]
            ),
        )
    )
    awaiting = next(e for e in pause_events if e.type == EventType.SESSION_AWAITING_USER_INPUT)
    input_id = awaiting.payload["input_id"]
    private_input_id = asyncio.run(private_event_for_public_event(store, awaiting)).payload[
        "input_id"
    ]

    resume_events = asyncio.run(
        _drain(
            app.resolve_user_input(
                UserInputResponse(session_id="s_deny", input_id=input_id, answer="ans")
            )
        )
    )
    assert resume_events[-1].type == EventType.SESSION_COMPLETED
    blocked = next(e for e in resume_events if e.type == EventType.TOOL_CALL_BLOCKED)
    private_blocked = asyncio.run(private_event_for_public_event(store, blocked))
    assert private_blocked.payload["input_id"] == private_input_id
    parts = {
        p.tool_call_id: p for p in _tool_result_parts(asyncio.run(store.load_transcript("s_deny")))
    }
    assert parts["call_1"].is_error is True
    assert parts["call_1"].content != "SHOULD_NOT_RUN"  # blocked, not executed
    assert parts["call_2"].content == "ans"


def test_resolve_user_input_rejects_structured_output_swap() -> None:
    # A resolver cannot swap the output-schema contract the paused run was created with: when
    # the run had a structured_output and the resolution supplies a DIFFERENT one, it is rejected
    # (mirrors the tool-approval contract check; a matching or absent spec is fine).
    app, _store = _build([("call_1", "ask_user", {"question": "q"})])
    pause = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_so",
                messages=[Message.text("user", "go")],
                structured_output=StructuredOutputSpec(
                    json_schema={"type": "object", "properties": {"a": {"type": "string"}}}
                ),
            ),
        )
    )
    awaiting = next(e for e in pause if e.type == EventType.SESSION_AWAITING_USER_INPUT)
    input_id = awaiting.payload["input_id"]
    with pytest.raises(ValueError, match="does not match the paused run contract"):
        asyncio.run(
            _drain(
                app.resolve_user_input(
                    UserInputResponse(
                        session_id="s_so",
                        input_id=input_id,
                        answer="a",
                        structured_output=StructuredOutputSpec(
                            json_schema={"type": "object", "properties": {"b": {"type": "number"}}}
                        ),
                    )
                )
            )
        )


def test_resolve_user_input_rejects_secret_structured_output_before_transition() -> None:
    secret = "user-input-schema-secret-canary"
    app, store = _build(
        [("call_1", "ask_user", {"question": "q"})],
        secret_redactor=SecretRedactor(secret),
    )
    pause = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_secret_structured_output",
                messages=[Message.text("user", "go")],
            ),
        )
    )
    input_id = next(
        event for event in pause if event.type == EventType.SESSION_AWAITING_USER_INPUT
    ).payload["input_id"]

    with pytest.raises(ValueError, match="workload secret"):
        asyncio.run(
            _drain(
                app.resolve_user_input(
                    UserInputResponse(
                        session_id="s_secret_structured_output",
                        input_id=input_id,
                        answer="a",
                        structured_output=StructuredOutputSpec(
                            json_schema={"type": "string", "const": secret},
                        ),
                    )
                )
            )
        )

    session = asyncio.run(store.load("s_secret_structured_output"))
    assert session is not None and session.status == SessionStatus.INTERRUPTED
    provider = app._get_registered_provider("fake").provider
    assert isinstance(provider, _ScriptedProvider)
    assert len(provider.requests) == 1


def test_resolve_user_input_rejects_native_structured_output_for_unsupported_provider() -> None:
    # The paused run had no spec, so supplying a NATIVE spec is execution-profile
    # drift and must be rejected before provider capability checks or status change.
    app, store = _build([("call_1", "ask_user", {"question": "q"})])
    pause = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_native",
                messages=[Message.text("user", "go")],
            ),
        )
    )
    awaiting = next(e for e in pause if e.type == EventType.SESSION_AWAITING_USER_INPUT)
    input_id = awaiting.payload["input_id"]

    with pytest.raises(ExecutionProfileMismatchError) as caught:
        asyncio.run(
            _drain(
                app.resolve_user_input(
                    UserInputResponse(
                        session_id="s_native",
                        input_id=input_id,
                        answer="a",
                        structured_output=StructuredOutputSpec(
                            json_schema={"type": "object"},
                            strategy=StructuredOutputStrategy.NATIVE,
                        ),
                    )
                )
            )
        )
    assert caught.value.changed_component_classes == (
        ExecutionProfileComponentClass.STRUCTURED_OUTPUT,
    )

    session = asyncio.run(store.load("s_native"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


def test_fork_of_paused_session_is_rejected() -> None:
    app, store = _build([("call_1", "ask_user", {"question": "q"})])
    asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_forksrc",
                messages=[Message.text("user", "go")],
            ),
        )
    )
    checkpoint = asyncio.run(store.load_checkpoint("s_forksrc"))
    with pytest.raises(RuntimeError, match="awaiting user input cannot be forked"):
        asyncio.run(
            _drain(
                app.fork_session(
                    ForkSessionRequest(source_session_id="s_forksrc", session_id="s_forkchild")
                )
            )
        )

    asyncio.run(store.update_status("s_forksrc", SessionStatus.FAILED))
    with pytest.raises(RuntimeError, match="awaiting user input cannot be forked"):
        asyncio.run(
            _drain(
                app.fork_session(
                    ForkSessionRequest(
                        source_session_id="s_forksrc",
                        session_id="s_forkchild_without_checkpoint",
                        copy_checkpoint=False,
                    )
                )
            )
        )
    assert asyncio.run(store.load("s_forkchild")) is None
    assert asyncio.run(store.load("s_forkchild_without_checkpoint")) is None
    assert asyncio.run(store.load_checkpoint("s_forksrc")) == checkpoint


class _FailOnceAppendStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    # Fails the next atomic user-input close before commit once armed.
    def __init__(self) -> None:
        super().__init__()
        self.armed = False

    async def publish_runtime_publication(self, session_id: str, **kwargs):
        if self.armed and kwargs["request"].kind == "user-input-close":
            self.armed = False
            raise RuntimeError("simulated append failure")
        return await super().publish_runtime_publication(session_id, **kwargs)


class _CountingTool(Tool):
    spec = ToolSpec(
        name="count",
        description="Counts executions.",
        input_schema={"type": "object", "properties": {}},
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls += 1
        return ToolResult(content=f"call-{self.calls}")


def test_retry_after_append_failure_does_not_re_execute_sibling() -> None:
    # Mixed round [count, ask_user]. First resolve runs `count`, then the atomic append fails ->
    # the session returns to INTERRUPTED (terminal event emitted). A retry must reuse the recorded
    # `count` outcome and NOT run it again.
    store = _FailOnceAppendStore()
    counting = _CountingTool()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(
        _ScriptedProvider([("call_1", "count", {}), ("call_2", "ask_user", {"question": "q"})]),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[UserInputTool(), counting],
    )
    pause = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant", session_id="s_retry", messages=[Message.text("user", "go")]
            ),
        )
    )
    input_id = next(e for e in pause if e.type == EventType.SESSION_AWAITING_USER_INPUT).payload[
        "input_id"
    ]

    store.armed = True  # fail the round-close append during the first resolve
    attempt1 = asyncio.run(
        _drain(
            app.resolve_user_input(
                UserInputResponse(session_id="s_retry", input_id=input_id, answer="a")
            )
        )
    )
    assert (
        attempt1[-1].type == EventType.SESSION_INTERRUPTED
    )  # append failed -> back to interrupted
    # The re-interrupt carries the failure so a caller can tell it apart from a fresh pause.
    assert attempt1[-1].payload.get("error_type")
    assert "error" in attempt1[-1].payload
    assert counting.calls == 1
    reloaded = asyncio.run(store.load("s_retry"))
    assert reloaded is not None and reloaded.status == SessionStatus.INTERRUPTED

    attempt2 = asyncio.run(
        _drain(
            app.resolve_user_input(
                UserInputResponse(session_id="s_retry", input_id=input_id, answer="a")
            )
        )
    )
    assert attempt2[-1].type == EventType.SESSION_COMPLETED
    assert counting.calls == 1  # reused recorded outcome; not re-executed


def test_retry_after_crashed_sibling_flags_manual_recovery_not_re_execute() -> None:
    # A sibling that STARTED on a prior resume but has no terminal event (a crash mid-tool) must
    # not be silently re-executed: the retry fails loudly with manual_recovery_required.
    store = InMemorySessionStore()
    counting = _CountingTool()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(
        _ScriptedProvider([("call_1", "count", {}), ("call_2", "ask_user", {"question": "q"})]),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[UserInputTool(), counting],
    )
    pause = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant", session_id="s_crash", messages=[Message.text("user", "go")]
            ),
        )
    )
    input_id = next(e for e in pause if e.type == EventType.SESSION_AWAITING_USER_INPUT).payload[
        "input_id"
    ]
    # Simulate a prior resume attempt that started `count` but crashed before a terminal event.
    asyncio.run(
        store.append_events(
            "s_crash",
            _crashed_user_input_resume_events(
                asyncio.run(private_events_for_public_events(store, pause)),
                session_id="s_crash",
                tool_call_id="call_1",
            ),
        )
    )

    events = asyncio.run(
        _drain(
            app.resolve_user_input(
                UserInputResponse(session_id="s_crash", input_id=input_id, answer="a")
            )
        )
    )
    assert events[-1].type == EventType.SESSION_INTERRUPTED
    assert events[-1].payload.get("manual_recovery_required") is True
    private_terminal = asyncio.run(private_event_for_public_event(store, events[-1]))
    assert private_terminal.payload.get("tool_call_id") == "call_1"
    assert counting.calls == 0  # guard fired before execution — no double-run
    reloaded = asyncio.run(store.load("s_crash"))
    assert reloaded is not None and reloaded.status == SessionStatus.INTERRUPTED


def test_recover_user_input_rejects_native_structured_output_for_unsupported_provider() -> None:
    # Manual recovery applies the same frozen-profile gate: a newly supplied
    # NATIVE contract is rejected before capability checks or status transition.
    store = InMemorySessionStore()
    counting = _CountingTool()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(
        _ScriptedProvider(
            [("call_1", "count", {}), ("call_2", "ask_user", {"question": "q"})],
            final_text="all done",
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[UserInputTool(), counting],
    )
    pause = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_rec_native",
                messages=[Message.text("user", "go")],
            ),
        )
    )
    awaiting = next(e for e in pause if e.type == EventType.SESSION_AWAITING_USER_INPUT)
    input_id = awaiting.payload["input_id"]
    asyncio.run(
        store.append_events(
            "s_rec_native",
            _crashed_user_input_resume_events(
                asyncio.run(private_events_for_public_events(store, pause)),
                session_id="s_rec_native",
                tool_call_id="call_1",
            ),
        )
    )
    stuck = asyncio.run(
        _drain(
            app.resolve_user_input(
                UserInputResponse(session_id="s_rec_native", input_id=input_id, answer="a")
            )
        )
    )
    assert stuck[-1].payload.get("manual_recovery_required") is True

    with pytest.raises(ExecutionProfileMismatchError) as caught:
        asyncio.run(
            _drain(
                app.recover_user_input(
                    UserInputRecoveryRequest(
                        session_id="s_rec_native",
                        input_id=input_id,
                        answer="a",
                        tool_call_id="call_1",
                        outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                        message="recovered externally",
                        structured_output=StructuredOutputSpec(
                            json_schema={"type": "object"},
                            strategy=StructuredOutputStrategy.NATIVE,
                        ),
                    )
                )
            )
        )
    assert caught.value.changed_component_classes == (
        ExecutionProfileComponentClass.STRUCTURED_OUTPUT,
    )

    session = asyncio.run(store.load("s_rec_native"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


def test_recover_user_input_rejects_secret_structured_output_before_transition() -> None:
    secret = "user-input-recovery-schema-secret-canary"
    store = InMemorySessionStore()
    counting = _CountingTool()
    provider = _ScriptedProvider(
        [("call_1", "count", {}), ("call_2", "ask_user", {"question": "q"})],
        final_text="all done",
    )
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        secret_redactor=SecretRedactor(secret),
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[UserInputTool(), counting],
    )
    session_id = "s_recover_secret_structured_output"
    pause = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
    )
    awaiting_input = next(
        event for event in pause if event.type == EventType.SESSION_AWAITING_USER_INPUT
    )
    input_id = awaiting_input.payload["input_id"]
    asyncio.run(
        store.append_events(
            session_id,
            _crashed_user_input_resume_events(
                asyncio.run(private_events_for_public_events(store, pause)),
                session_id=session_id,
                tool_call_id="call_1",
            ),
        )
    )
    stuck = asyncio.run(
        _drain(
            app.resolve_user_input(
                UserInputResponse(session_id=session_id, input_id=input_id, answer="a")
            )
        )
    )
    assert stuck[-1].payload.get("manual_recovery_required") is True

    with pytest.raises(ValueError, match="workload secret"):
        asyncio.run(
            _drain(
                app.recover_user_input(
                    UserInputRecoveryRequest(
                        session_id=session_id,
                        input_id=input_id,
                        answer="a",
                        tool_call_id="call_1",
                        outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                        message="recovered externally",
                        structured_output=StructuredOutputSpec(
                            json_schema={"type": "string", "const": secret},
                        ),
                    )
                )
            )
        )

    session = asyncio.run(store.load(session_id))
    assert session is not None and session.status == SessionStatus.INTERRUPTED
    assert len(provider.requests) == 1


@pytest.mark.parametrize("dynamic_scope", [False, True], ids=["static", "dynamic"])
def test_recover_user_input_supplies_outcome_and_completes(
    dynamic_scope: bool,
) -> None:
    # After a crashed sibling leaves the round on manual_recovery_required, recover_user_input
    # supplies the missing outcome; the round finishes without re-running the sibling, and the
    # re-supplied answer is injected as the ask_user result (it was unrecorded before the crash).
    store = InMemorySessionStore()
    counting = _CountingTool()
    session_id = f"s_rec_{'dynamic' if dynamic_scope else 'static'}"
    recovery_message = "manual-recovery-secret-canary" if dynamic_scope else "recovered externally"

    class ObserveManualRecovery(RuntimeHook):
        def __init__(self) -> None:
            self.arguments: list[dict] = []

        async def after_tool_call(self, context: ToolCallHookContext) -> None:
            self.arguments.append(context.arguments)

    observer = ObserveManualRecovery()
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        runtime_hooks=[observer],
    )
    provider = _ScriptedProvider(
        [
            ("call_1", "count", {"private": "pending"}),
            ("call_2", "ask_user", {"question": "q"}),
        ],
        final_text="all done",
    )
    app.register_provider(provider, default=True)
    if dynamic_scope:
        app.register_environment(
            Environment(
                EnvironmentSpec(name="dynamic"),
                vault=StaticVault({"api_key": recovery_message}),
            ),
            default=True,
        )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[UserInputTool(), counting],
    )
    pause = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant", session_id=session_id, messages=[Message.text("user", "go")]
            ),
        )
    )
    awaiting = next(e for e in pause if e.type == EventType.SESSION_AWAITING_USER_INPUT)
    input_id = awaiting.payload["input_id"]
    private_input_id = asyncio.run(private_event_for_public_event(store, awaiting)).payload[
        "input_id"
    ]
    # Simulate a prior resume that started `count` but crashed before a terminal event.
    asyncio.run(
        store.append_events(
            session_id,
            _crashed_user_input_resume_events(
                asyncio.run(private_events_for_public_events(store, pause)),
                session_id=session_id,
                tool_call_id="call_1",
            ),
        )
    )
    stuck = asyncio.run(
        _drain(
            app.resolve_user_input(
                UserInputResponse(
                    session_id=session_id,
                    input_id=input_id,
                    answer="a",
                    structured={"answer_detail": "safe"},
                    metadata={"provided": recovery_message},
                )
            )
        )
    )
    assert stuck[-1].payload.get("manual_recovery_required") is True

    recovered = asyncio.run(
        _drain(
            app.recover_user_input(
                UserInputRecoveryRequest(
                    session_id=session_id,
                    input_id=input_id,
                    answer="a",
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message=recovery_message,
                    # ``structured`` belongs to the re-supplied ask_user answer as well as
                    # the recovered sibling result. Keep the answer unrelated so this test
                    # isolates the manual-recovery publication boundary.
                    structured={"answer_detail": "safe"},
                    reason=recovery_message,
                    metadata={"provided": recovery_message},
                )
            )
        )
    )
    recovered_tool_event = next(
        event
        for event in recovered
        if event.type == EventType.TOOL_CALL_COMPLETED
        and event.payload.get("manual_recovery") is True
    )
    private_recovered_tool_event = asyncio.run(
        private_event_for_public_event(store, recovered_tool_event)
    )
    assert private_recovered_tool_event.payload[
        "idempotency_key"
    ] == tool_execution.tool_idempotency_key(
        session_id=session_id,
        tool_round_id=private_recovered_tool_event.payload["tool_round_id"],
        tool_call_id="call_1",
        pause_id=private_input_id,
    )
    assert private_recovered_tool_event.payload["reason"] == (
        None if dynamic_scope else recovery_message
    )
    assert private_recovered_tool_event.payload["metadata"] == (
        {} if dynamic_scope else {"provided": recovery_message}
    )
    assert recovered[-1].type == EventType.SESSION_COMPLETED
    assert counting.calls == 0  # the recovered tool was never re-executed
    assert observer.arguments
    assert all(arguments == {} for arguments in observer.arguments)
    checkpoint = asyncio.run(store.load_checkpoint(session_id))
    assert checkpoint is not None
    assert "pending_user_input" not in checkpoint
    parts = _tool_result_parts(asyncio.run(store.load_transcript(session_id)))
    results = {part.tool_call_id: part.content for part in parts}
    if dynamic_scope:
        assert results["call_1"] == (
            "Externally verified tool output is unavailable because the invocation "
            "secret scope could not be reconstructed."
        )
        assert recovery_message not in repr(recovered)
        assert recovery_message not in repr(asyncio.run(store.load_events(session_id)))
        assert recovery_message not in repr(asyncio.run(store.load_transcript(session_id)))
        assert recovery_message not in repr(provider.requests[-1].messages)
    else:
        assert results["call_1"] == "recovered externally"
    assert results["call_2"] == "a"  # ask_user answer injected on continuation


def test_recover_user_input_reconciles_ambiguous_append_acknowledgement() -> None:
    class AmbiguousRecoveryAppendStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.failed_recovery_ack = False

        async def append_events(self, session_id: str, events: list[Event]) -> None:
            manual_recovery = any(event.payload.get("manual_recovery") is True for event in events)
            await super().append_events(session_id, events)
            if manual_recovery and not self.failed_recovery_ack:
                self.failed_recovery_ack = True
                raise RuntimeError("user-input recovery commit acknowledgement lost")

    async def scenario() -> None:
        session_id = "s_rec_ambiguous_append"
        store = AmbiguousRecoveryAppendStore()
        counting = _CountingTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            _ScriptedProvider(
                [("call_1", "count", {}), ("call_2", "ask_user", {"question": "q"})],
                final_text="all done",
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool(), counting],
        )
        paused = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
        input_id = next(
            event for event in paused if event.type == EventType.SESSION_AWAITING_USER_INPUT
        ).payload["input_id"]
        await store.append_events(
            session_id,
            _crashed_user_input_resume_events(
                await private_events_for_public_events(store, paused),
                session_id=session_id,
                tool_call_id="call_1",
            ),
        )
        stuck = await _drain(
            app.resolve_user_input(
                UserInputResponse(session_id=session_id, input_id=input_id, answer="a")
            )
        )
        assert stuck[-1].payload.get("manual_recovery_required") is True

        recovery = await _drain(
            app.recover_user_input(
                UserInputRecoveryRequest(
                    session_id=session_id,
                    input_id=input_id,
                    answer="a",
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="recovered externally",
                )
            )
        )
        session = await store.load(session_id)
        assert session is not None and session.status == SessionStatus.INTERRUPTED
        assert recovery[-1].type == EventType.SESSION_INTERRUPTED
        assert recovery[-1].payload["manual_recovery_persisted"] is True
        persisted = await store.load_events(session_id)
        recovered = [
            event
            for event in persisted
            if event.payload.get("manual_recovery") is True
            and event.payload.get("tool_call_id") == "call_1"
        ]
        assert len(recovered) == 1

        resumed = await _drain(
            app.resolve_user_input(
                UserInputResponse(session_id=session_id, input_id=input_id, answer="a")
            )
        )
        assert resumed[-1].type == EventType.SESSION_COMPLETED
        assert counting.calls == 0

    asyncio.run(scenario())


def test_recover_user_input_claim_rejects_conflicting_decision_after_lost_acknowledgement() -> None:
    async def scenario() -> None:
        session_id = "s_recovery_claim_lost_ack_conflict"
        store = _BlockingCommittedRunningTransitionStore()
        counting = _CountingTool()
        provider = _ScriptedProvider(
            [("call_1", "count", {}), ("call_2", "ask_user", {"question": "q"})],
            final_text="all done",
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool(), counting],
        )
        paused = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
        input_id = next(
            event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
        ).payload["input_id"]
        await store.append_events(
            session_id,
            _crashed_user_input_resume_events(
                await private_events_for_public_events(store, paused),
                session_id=session_id,
                tool_call_id="call_1",
            ),
        )
        answer_metadata = {"source": "lost-ack-regression"}
        stuck = await _drain(
            app.resolve_user_input(
                UserInputResponse(
                    session_id=session_id,
                    input_id=input_id,
                    answer="a",
                    metadata=answer_metadata,
                )
            )
        )
        assert stuck[-1].payload.get("manual_recovery_required") is True

        request = UserInputRecoveryRequest(
            session_id=session_id,
            input_id=input_id,
            answer="a",
            tool_call_id="call_1",
            outcome=ToolApprovalRecoveryOutcome.COMPLETED,
            message="verified completed",
            reason="operator inspected the external system",
            metadata=answer_metadata,
        )
        store.transition_committed = asyncio.Event()
        store.finish_transition = asyncio.Event()
        store.block_next_running_transition = True
        recovering = asyncio.create_task(_drain(app.recover_user_input(request)))
        await asyncio.wait_for(store.transition_committed.wait(), timeout=5)

        claimed_session = await store.load(session_id)
        claimed_checkpoint = await store.load_checkpoint(session_id)
        assert claimed_session is not None
        assert claimed_session.status is SessionStatus.RUNNING
        assert claimed_checkpoint is not None
        claimed_intent = claimed_checkpoint["user_input_resolution_intent"]
        private_request = request.model_copy(
            update={"input_id": claimed_checkpoint["pending_user_input"]["input_id"]}
        )
        assert claimed_intent["answer_request_digest"] == user_input_answer_request_digest(
            private_request
        )
        assert claimed_intent["resolution_stage"] == "manual-recovery"
        assert claimed_intent["resolution_request_digest"] == user_input_resolution_request_digest(
            private_request
        )

        events_before_conflicts = await store.load_events(session_id)
        checkpoint_before_conflicts = await store.load_checkpoint(session_id)
        provider_requests_before_conflicts = list(provider.requests)
        conflicting_requests = (
            request.model_copy(update={"message": "verified failed instead"}),
            request.model_copy(update={"outcome": ToolApprovalRecoveryOutcome.FAILED}),
            request.model_copy(update={"reason": "different operator evidence"}),
        )
        for conflicting_request in conflicting_requests:
            assert user_input_answer_request_digest(
                conflicting_request
            ) == user_input_answer_request_digest(request)
            assert user_input_resolution_request_digest(
                conflicting_request
            ) != user_input_resolution_request_digest(request)
            with pytest.raises(
                SessionRuntimePublicationConflict,
                match="different resolution request",
            ):
                await _drain(app.recover_user_input(conflicting_request))
            assert await store.load_checkpoint(session_id) == checkpoint_before_conflicts
            assert await store.load_events(session_id) == events_before_conflicts
            assert provider.requests == provider_requests_before_conflicts

        store.finish_transition.set()
        recovered = await asyncio.wait_for(recovering, timeout=10)
        assert recovered[-1].type is EventType.SESSION_COMPLETED
        assert counting.calls == 0

    asyncio.run(scenario())


def test_operator_interrupt_supersedes_user_input_manual_recovery_before_execution() -> None:
    async def scenario() -> None:
        session_id = "s_recovery_claim_operator_supersession"
        store = _BlockingCommittedRunningTransitionStore()
        counting = _CountingTool()
        provider = _ScriptedProvider(
            [("call_1", "count", {}), ("call_2", "ask_user", {"question": "q"})],
            final_text="must not dispatch",
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool(), counting],
        )
        paused = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
        input_id = next(
            event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
        ).payload["input_id"]
        await store.append_events(
            session_id,
            _crashed_user_input_resume_events(
                await private_events_for_public_events(store, paused),
                session_id=session_id,
                tool_call_id="call_1",
            ),
        )
        answer_metadata = {"source": "operator-supersession-regression"}
        stuck = await _drain(
            app.resolve_user_input(
                UserInputResponse(
                    session_id=session_id,
                    input_id=input_id,
                    answer="a",
                    metadata=answer_metadata,
                )
            )
        )
        assert stuck[-1].payload.get("manual_recovery_required") is True
        resumed_before_recovery = sum(
            event.type is EventType.SESSION_RESUMED for event in await store.load_events(session_id)
        )

        recovery_request = UserInputRecoveryRequest(
            session_id=session_id,
            input_id=input_id,
            answer="a",
            tool_call_id="call_1",
            outcome=ToolApprovalRecoveryOutcome.COMPLETED,
            message="verified completed",
            metadata=answer_metadata,
        )
        store.transition_committed = asyncio.Event()
        store.finish_transition = asyncio.Event()
        store.block_next_running_transition = True
        recovering = asyncio.create_task(_drain(app.recover_user_input(recovery_request)))
        await asyncio.wait_for(store.transition_committed.wait(), timeout=5)

        claimed_checkpoint = await store.load_checkpoint(session_id)
        assert claimed_checkpoint is not None
        assert claimed_checkpoint["user_input_resolution_intent"]["execution_state"] == "claimed"

        interrupt_app = CayuApp(session_store=store, enable_logging=False)
        interrupt_app.register_provider(provider, default=True)
        interrupt_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool(), counting],
        )
        with pytest.raises(TimeoutError, match="still finalizing"):
            _ = [
                event
                async for event in interrupt_app.interrupt_session(
                    InterruptSessionRequest(
                        session_id=session_id,
                        reason="operator supersedes manual recovery claim",
                    )
                )
            ]

        store.finish_transition.set()
        recovery_events = await asyncio.wait_for(recovering, timeout=10)
        assert recovery_events[-1].type is EventType.SESSION_INTERRUPTED
        assert recovery_events[-1].payload["interruption_type"] == "operator_requested"
        assert counting.calls == 0
        assert len(provider.requests) == 1
        durable_events = await store.load_events(session_id)
        assert not any(event.payload.get("manual_recovery") is True for event in durable_events)
        assert (
            sum(event.type is EventType.SESSION_RESUMED for event in durable_events)
            == resumed_before_recovery
        )
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "pending_user_input" not in checkpoint
        assert "user_input_resolution_intent" not in checkpoint

    asyncio.run(scenario())


def test_operator_interrupt_cannot_supersede_executing_user_input_manual_recovery() -> None:
    async def scenario() -> None:
        session_id = "s_executing_recovery_claim_rejects_operator_supersession"
        store = _BlockingCommittedRunningTransitionStore()
        counting = _CountingTool()
        provider = _ScriptedProvider(
            [("call_1", "count", {}), ("call_2", "ask_user", {"question": "q"})],
            final_text="continued after verified recovery",
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool(), counting],
        )
        paused = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
        input_id = next(
            event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
        ).payload["input_id"]
        await store.append_events(
            session_id,
            _crashed_user_input_resume_events(
                await private_events_for_public_events(store, paused),
                session_id=session_id,
                tool_call_id="call_1",
            ),
        )
        metadata = {"source": "executing-recovery-supersession-regression"}
        stuck = await _drain(
            app.resolve_user_input(
                UserInputResponse(
                    session_id=session_id,
                    input_id=input_id,
                    answer="a",
                    metadata=metadata,
                )
            )
        )
        assert stuck[-1].payload.get("manual_recovery_required") is True

        request = UserInputRecoveryRequest(
            session_id=session_id,
            input_id=input_id,
            answer="a",
            tool_call_id="call_1",
            outcome=ToolApprovalRecoveryOutcome.COMPLETED,
            message="verified completed",
            metadata=metadata,
        )
        store.execution_admission_committed = asyncio.Event()
        store.finish_execution_admission = asyncio.Event()
        store.block_next_execution_admission = True
        recovering = asyncio.create_task(_drain(app.recover_user_input(request)))
        await asyncio.wait_for(store.execution_admission_committed.wait(), timeout=5)

        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert checkpoint["user_input_resolution_intent"]["execution_state"] == "executing"

        interrupt_app = CayuApp(session_store=store, enable_logging=False)
        interrupt_app.register_provider(provider, default=True)
        interrupt_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool(), counting],
        )
        with pytest.raises(
            SessionRuntimePublicationConflict,
            match="already executing and cannot be superseded",
        ):
            await _drain(
                interrupt_app.interrupt_session(
                    InterruptSessionRequest(
                        session_id=session_id,
                        reason="cannot supersede executing manual recovery",
                    )
                )
            )

        assert not any(
            event.type is EventType.SESSION_INTERRUPTED
            and event.payload.get("interruption_type") == "operator_requested"
            for event in await store.load_events(session_id)
        )
        store.finish_execution_admission.set()
        recovered = await asyncio.wait_for(recovering, timeout=10)
        assert recovered[-1].type is EventType.SESSION_COMPLETED
        assert counting.calls == 0
        assert len(provider.requests) == 2
        assert any(
            event.payload.get("manual_recovery") is True
            for event in await store.load_events(session_id)
        )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "grouped_cancellation",
    [False, True],
    ids=["ordinary-error", "grouped-cancellation"],
)
def test_recover_user_input_post_persist_fanout_failure_stays_resumable(
    grouped_cancellation: bool,
) -> None:
    async def scenario() -> None:
        failure_kind = "grouped" if grouped_cancellation else "ordinary"
        session_id = f"s_rec_post_persist_failure_{failure_kind}"
        store = InMemorySessionStore()
        counting = _CountingTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            _ScriptedProvider(
                [("call_1", "count", {}), ("call_2", "ask_user", {"question": "q"})],
                final_text="all done",
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool(), counting],
        )
        paused = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
        input_id = next(
            event for event in paused if event.type == EventType.SESSION_AWAITING_USER_INPUT
        ).payload["input_id"]
        await store.append_events(
            session_id,
            _crashed_user_input_resume_events(
                await private_events_for_public_events(store, paused),
                session_id=session_id,
                tool_call_id="call_1",
            ),
        )
        stuck = await _drain(
            app.resolve_user_input(
                UserInputResponse(session_id=session_id, input_id=input_id, answer="a")
            )
        )
        assert stuck[-1].payload.get("manual_recovery_required") is True

        original_fan_out = app._event_writer.fan_out_persisted
        failed = False
        fan_out_failure: BaseException = (
            BaseExceptionGroup(
                "user-input recovery fan-out cancelled and failed",
                [asyncio.CancelledError("fan-out cancelled"), RuntimeError("fan-out failed")],
            )
            if grouped_cancellation
            else RuntimeError("user-input recovery fan-out unavailable")
        )

        async def fail_recovery_fan_out(events: list[Event]) -> list[Event]:
            nonlocal failed
            if not failed and any(event.payload.get("manual_recovery") is True for event in events):
                failed = True
                raise fan_out_failure
            return await original_fan_out(events)

        app._event_writer.fan_out_persisted = fail_recovery_fan_out
        recovery_request = UserInputRecoveryRequest(
            session_id=session_id,
            input_id=input_id,
            answer="a",
            tool_call_id="call_1",
            outcome=ToolApprovalRecoveryOutcome.COMPLETED,
            message="recovered externally",
        )
        recovery: list[Event] = []
        if grouped_cancellation:
            with pytest.raises(BaseExceptionGroup) as raised:
                await _drain(app.recover_user_input(recovery_request))
            assert raised.value is fan_out_failure
        else:
            recovery = await _drain(app.recover_user_input(recovery_request))
        session = await store.load(session_id)
        assert session is not None and session.status == SessionStatus.INTERRUPTED
        persisted = await store.load_events(session_id)
        terminal = [event for event in persisted if event.type == EventType.SESSION_INTERRUPTED][-1]
        if grouped_cancellation:
            assert terminal.payload.get("abandoned") is not True
        else:
            sequence = public_event_sequence(recovery[-1].id)
            assert sequence is not None
            records = await store.query_events(
                EventQuery(session_id=session_id, after_sequence=sequence - 1, limit=1)
            )
            assert len(records) == 1
            assert records[0].sequence == sequence
            assert records[0].event.id == terminal.id
            assert terminal.payload["manual_recovery_persisted"] is True
        assert (
            len(
                [
                    event
                    for event in persisted
                    if event.payload.get("manual_recovery") is True
                    and event.payload.get("tool_call_id") == "call_1"
                ]
            )
            == 1
        )

        resumed = await _drain(
            app.resolve_user_input(
                UserInputResponse(session_id=session_id, input_id=input_id, answer="a")
            )
        )
        assert resumed[-1].type == EventType.SESSION_COMPLETED
        assert counting.calls == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("corruption", ["missing", "conflicting"])
def test_resolve_user_input_requires_exact_manual_recovery_terminal_authority(
    corruption: str,
) -> None:
    class CorruptManualRecoveryStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def append_events(self, session_id: str, events: list[Event]) -> None:
            corrupted: list[Event] = []
            for event in events:
                if event.payload.get("manual_recovery") is not True:
                    corrupted.append(event)
                    continue
                payload = dict(event.payload)
                if corruption == "missing":
                    payload.pop("resolution_request_digest")
                else:
                    payload["resolution_request_digest"] = "f" * 64
                corrupted.append(event.model_copy(update={"payload": payload}))
            await super().append_events(session_id, corrupted)

    async def scenario() -> None:
        session_id = f"s_manual_recovery_authority_{corruption}"
        store = CorruptManualRecoveryStore()
        app = CayuApp(session_store=store, enable_logging=False)
        provider = _ScriptedProvider(
            [("call_1", "count", {}), ("call_2", "ask_user", {"question": "q"})],
            final_text="all done",
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool(), _CountingTool()],
        )
        paused = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
        input_id = next(
            event for event in paused if event.type == EventType.SESSION_AWAITING_USER_INPUT
        ).payload["input_id"]
        await store.append_events(
            session_id,
            _crashed_user_input_resume_events(
                await private_events_for_public_events(store, paused),
                session_id=session_id,
                tool_call_id="call_1",
            ),
        )
        stuck = await _drain(
            app.resolve_user_input(
                UserInputResponse(session_id=session_id, input_id=input_id, answer="a")
            )
        )
        assert stuck[-1].payload["manual_recovery_required"] is True

        original_fan_out = app._event_writer.fan_out_persisted
        failed = False

        async def fail_recovery_fan_out(events: list[Event]) -> list[Event]:
            nonlocal failed
            if not failed and any(event.payload.get("manual_recovery") is True for event in events):
                failed = True
                raise RuntimeError("manual recovery fan-out failed")
            return await original_fan_out(events)

        app._event_writer.fan_out_persisted = fail_recovery_fan_out
        recovered = await _drain(
            app.recover_user_input(
                UserInputRecoveryRequest(
                    session_id=session_id,
                    input_id=input_id,
                    answer="a",
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="recovered externally",
                )
            )
        )
        assert recovered[-1].payload["manual_recovery_persisted"] is True

        checkpoint_before = await store.load_checkpoint(session_id)
        events_before = await store.load_events(session_id)
        provider_requests_before = list(provider.requests)
        with pytest.raises(
            SessionRuntimePublicationConflict,
            match="different resolution request",
        ):
            await _drain(
                app.resolve_user_input(
                    UserInputResponse(session_id=session_id, input_id=input_id, answer="a")
                )
            )
        assert await store.load_checkpoint(session_id) == checkpoint_before
        assert await store.load_events(session_id) == events_before
        assert provider.requests == provider_requests_before

    asyncio.run(scenario())


def test_recover_user_input_post_persist_cleanup_failure_is_not_suppressed() -> None:
    async def scenario() -> None:
        session_id = "s_rec_post_persist_cleanup_failure"
        store = _FailingReleaseBeforeCleanupStore()
        counting = _CountingTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            _ScriptedProvider(
                [("call_1", "count", {}), ("call_2", "ask_user", {"question": "q"})],
                final_text="all done",
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool(), counting],
        )
        paused = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
        input_id = next(
            event for event in paused if event.type == EventType.SESSION_AWAITING_USER_INPUT
        ).payload["input_id"]
        await store.append_events(
            session_id,
            _crashed_user_input_resume_events(
                await private_events_for_public_events(store, paused),
                session_id=session_id,
                tool_call_id="call_1",
            ),
        )
        stuck = await _drain(
            app.resolve_user_input(
                UserInputResponse(session_id=session_id, input_id=input_id, answer="a")
            )
        )
        assert stuck[-1].payload.get("manual_recovery_required") is True

        original_fan_out = app._event_writer.fan_out_persisted
        failed = False

        async def fail_recovery_fan_out(events: list[Event]) -> list[Event]:
            nonlocal failed
            if not failed and any(event.payload.get("manual_recovery") is True for event in events):
                failed = True
                raise RuntimeError("user-input recovery fan-out unavailable")
            return await original_fan_out(events)

        app._event_writer.fan_out_persisted = fail_recovery_fan_out
        store.fail_next_release = True
        with pytest.raises(
            RuntimeError,
            match="run fence release unavailable before cleanup",
        ):
            await _drain(
                app.recover_user_input(
                    UserInputRecoveryRequest(
                        session_id=session_id,
                        input_id=input_id,
                        answer="a",
                        tool_call_id="call_1",
                        outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                        message="recovered externally",
                    )
                )
            )

        session = await store.load(session_id)
        assert session is not None and session.status == SessionStatus.INTERRUPTED
        persisted = await store.load_events(session_id)
        assert persisted[-1].type == EventType.SESSION_INTERRUPTED
        assert persisted[-1].payload["manual_recovery_persisted"] is True
        assert counting.calls == 0

    asyncio.run(scenario())


def test_recover_user_input_closes_continuation_before_aclose_returns() -> None:
    async def run() -> tuple[Event, SessionStatus, int, bool]:
        session_id = "s_recovery_stream_closed"
        store = _RecordingReleaseStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            _ScriptedProvider(
                [("call_1", "count", {}), ("call_2", "ask_user", {"question": "q"})],
                final_text="all done",
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool(), _CountingTool()],
        )
        pause = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
        input_id = next(
            event for event in pause if event.type == EventType.SESSION_AWAITING_USER_INPUT
        ).payload["input_id"]
        await store.append_events(
            session_id,
            _crashed_user_input_resume_events(
                await private_events_for_public_events(store, pause),
                session_id=session_id,
                tool_call_id="call_1",
            ),
        )
        stuck = await _drain(
            app.resolve_user_input(
                UserInputResponse(session_id=session_id, input_id=input_id, answer="a")
            )
        )
        assert stuck[-1].payload.get("manual_recovery_required") is True

        releases_before = store.release_calls.get(session_id, 0)
        stream = app.recover_user_input(
            UserInputRecoveryRequest(
                session_id=session_id,
                input_id=input_id,
                answer="a",
                tool_call_id="call_1",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="recovered externally",
            )
        )
        while True:
            boundary_event = await anext(stream)
            if boundary_event.type == EventType.MODEL_STARTED:
                break
        assert app._session_control.has_active_tasks(session_id) is True
        await stream.aclose()
        release_delta = store.release_calls.get(session_id, 0) - releases_before
        has_active_tasks = app._session_control.has_active_tasks(session_id)
        session = await store.load(session_id)
        assert session is not None
        return boundary_event, session.status, release_delta, has_active_tasks

    boundary_event, status, release_delta, has_active_tasks = asyncio.run(run())

    assert boundary_event.type == EventType.MODEL_STARTED
    assert status == SessionStatus.INTERRUPTED
    assert release_delta == 1
    assert has_active_tasks is False


def test_recover_user_input_task_cancellation_finalizes_continuation() -> None:
    async def run() -> None:
        session_id = "s_recovery_task_cancelled"
        store = _RecordingReleaseStore()
        provider = _BlockingContinuationProvider(
            [("call_count", "count", {}), ("call_input", "ask_user", {"question": "q"})]
        )
        provider.continuation_started = asyncio.Event()
        provider.never_complete = asyncio.Event()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool(), _CountingTool()],
        )
        pause = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
        input_id = next(
            event for event in pause if event.type == EventType.SESSION_AWAITING_USER_INPUT
        ).payload["input_id"]
        await store.append_events(
            session_id,
            _crashed_user_input_resume_events(
                await private_events_for_public_events(store, pause),
                session_id=session_id,
                tool_call_id="call_count",
            ),
        )
        stuck = await _drain(
            app.resolve_user_input(
                UserInputResponse(session_id=session_id, input_id=input_id, answer="yes")
            )
        )
        assert stuck[-1].payload["manual_recovery_required"] is True

        releases_before = store.release_calls[session_id]
        recovery_task = asyncio.create_task(
            _drain(
                app.recover_user_input(
                    UserInputRecoveryRequest(
                        session_id=session_id,
                        input_id=input_id,
                        answer="yes",
                        tool_call_id="call_count",
                        outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                        message="count completed externally",
                    )
                )
            )
        )
        await asyncio.wait_for(provider.continuation_started.wait(), timeout=5)
        assert recovery_task.cancelling() == 0
        recovery_task.cancel("cancel user-input recovery")
        assert recovery_task.cancelling() == 1
        try:
            await recovery_task
        except asyncio.CancelledError as cancellation:
            assert cancellation.args == ("Provider operation cancelled",)
        else:
            pytest.fail("User-input recovery did not preserve task cancellation.")

        assert recovery_task.cancelled() is True
        assert recovery_task.cancelling() == 1
        session = await store.load(session_id)
        assert session is not None
        assert session.status == SessionStatus.INTERRUPTED
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "pending_user_input" not in checkpoint
        events = await store.load_events(session_id)
        assert events[-1].type == EventType.SESSION_INTERRUPTED
        assert events[-1].payload["abandoned"] is True
        assert store.release_calls[session_id] - releases_before == 1
        assert app._session_control.has_active_tasks(session_id) is False

    asyncio.run(run())


def test_recover_user_input_rejects_tool_without_started_event() -> None:
    # A tool_call_id that never started is not a valid recovery target.
    app, _store = _build([("call_1", "ask_user", {"question": "q"})])
    asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant", session_id="s_rec2", messages=[Message.text("user", "go")]
            ),
        )
    )
    checkpoint = asyncio.run(_store.load_checkpoint("s_rec2"))
    input_id = checkpoint["pending_user_input"]["input_id"]
    with pytest.raises(RuntimeError, match="requires a recorded tool.call.started"):
        asyncio.run(
            _drain(
                app.recover_user_input(
                    UserInputRecoveryRequest(
                        session_id="s_rec2",
                        input_id=input_id,
                        answer="a",
                        tool_call_id="call_1",
                        outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                        message="x",
                    )
                )
            )
        )


class _TwoRoundProvider(ModelProvider):
    """Round 1: [count(call_1), echo(call_2)]; round 2: [count(call_1), ask_user(call_2)] — the
    same tool-call ids reused across rounds (ids are only unique within one assistant message)."""

    name = "fake"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        n = len(self.requests)
        if n == 1:
            yield ModelStreamEvent.tool_call(id="call_1", name="count", arguments={})
            yield ModelStreamEvent.tool_call(id="call_2", name="echo", arguments={"text": "round1"})
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
        elif n == 2:
            yield ModelStreamEvent.tool_call(id="call_1", name="count", arguments={})
            yield ModelStreamEvent.tool_call(
                id="call_2", name="ask_user", arguments={"question": "q"}
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
        else:
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})


def test_resume_does_not_reuse_a_prior_rounds_outcomes_by_reused_id() -> None:
    # Regression: the resume ledger must scope to this pause's resume window, not match a prior
    # round's terminal events that reuse the same tool_call_id — otherwise the sibling never runs
    # and the answer is replaced by a stale result.
    store = InMemorySessionStore()
    counting = _CountingTool()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(_TwoRoundProvider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[UserInputTool(), _EchoTool(), counting],
    )
    pause = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant", session_id="s_reuse", messages=[Message.text("user", "go")]
            ),
        )
    )
    assert counting.calls == 1  # round 1 ran count once
    input_id = next(e for e in pause if e.type == EventType.SESSION_AWAITING_USER_INPUT).payload[
        "input_id"
    ]
    resume = asyncio.run(
        _drain(
            app.resolve_user_input(
                UserInputResponse(session_id="s_reuse", input_id=input_id, answer="MY-ANSWER")
            )
        )
    )
    assert resume[-1].type == EventType.SESSION_COMPLETED
    assert counting.calls == 2  # round-2 count ran fresh; round-1 outcome was NOT reused
    transcript = asyncio.run(store.load_transcript("s_reuse"))
    last_tool_message = [m for m in transcript if m.role == "tool"][-1]  # round 2's results
    parts = {p.tool_call_id: p for p in last_tool_message.content if isinstance(p, ToolResultPart)}
    # call_2 in round 2 is ask_user — its result is the injected answer, not round 1's echo "round1".
    assert parts["call_2"].content == "MY-ANSWER"
    assert parts["call_1"].content == "call-2"  # count's second execution


def test_worker_recovery_preserves_pending_user_input() -> None:
    # A crash with status still RUNNING and a pending_user_input checkpoint must be recovered as
    # user_input_required with the question payload (discoverable via the documented contract),
    # not as an opaque runtime_interrupted with no payload/id.
    app, store = _build([("call_1", "ask_user", {"question": "which env?", "options": ["dev"]})])
    pause = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_crashrec",
                messages=[Message.text("user", "go")],
            ),
        )
    )
    awaiting = next(e for e in pause if e.type == EventType.SESSION_AWAITING_USER_INPUT)
    private_input_id = asyncio.run(private_event_for_public_event(store, awaiting)).payload[
        "input_id"
    ]
    # Simulate the crash window with the profile rebound in the same running-epoch claim.
    asyncio.run(rebind_test_invocation(store, "s_crashrec"))
    result = asyncio.run(
        app.recover_incomplete_session(IncompleteSessionRecoveryRequest(session_id="s_crashrec"))
    )
    assert IncompleteSessionRecoveryAction.PENDING_USER_INPUT in result.actions
    assert result.pending_user_input_id is not None
    assert (
        asyncio.run(
            app._resolve_public_action_linkage(
                session_id="s_crashrec",
                value=result.pending_user_input_id,
                field_name="input_id",
            )
        )
        == private_input_id
    )
    interrupted = [e for e in result.events if e.type == EventType.SESSION_INTERRUPTED]
    assert interrupted and interrupted[-1].payload["interruption_type"] == "user_input_required"
    assert interrupted[-1].payload["user_input"]["question"] == "which env?"
    assert asyncio.run(store.load("s_crashrec")).status == SessionStatus.INTERRUPTED


def test_worker_recovery_rejects_pause_without_exact_open_receipt_as_ambiguous() -> None:
    class MissingOpenReceiptStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.hide_open_receipt = False

        async def load_runtime_publication_receipt(
            self,
            session_id: str,
            publication_id: str,
        ):
            if self.hide_open_receipt and publication_id.startswith("user-input-open:"):
                return None
            return await super().load_runtime_publication_receipt(
                session_id,
                publication_id,
            )

    async def run() -> None:
        session_id = "s_ambiguous_user_input_open_receipt"
        store = MissingOpenReceiptStore()
        app, _ = _build(
            [("call_1", "ask_user", {"question": "which env?"})],
            store=store,
        )
        await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
        await rebind_test_invocation(store, session_id)
        store.hide_open_receipt = True
        checkpoint_before = await store.load_checkpoint(session_id)
        transcript_before = await store.load_transcript(session_id)
        events_before = await store.load_events(session_id)

        with pytest.raises(
            SessionRuntimePublicationConflict,
            match="recovery authority is ambiguous",
        ):
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )

        checkpoint_after = await store.load_checkpoint(session_id)
        assert checkpoint_after is not None
        assert checkpoint_after["pending_user_input"] == checkpoint_before["pending_user_input"]
        assert "user_input_resolution_intent" not in checkpoint_after
        assert "user_input_supersession_intent" not in checkpoint_after
        assert await store.load_transcript(session_id) == transcript_before
        events_after = await store.load_events(session_id)
        assert events_after[: len(events_before)] == events_before
        assert all(
            event.type is EventType.SESSION_RUN_FENCED
            for event in events_after[len(events_before) :]
        )

    asyncio.run(run())


@pytest.mark.parametrize(
    "corruption",
    [
        pytest.param("nested-arguments", id="nested-arguments"),
        pytest.param("target-arguments", id="top-level-target"),
    ],
)
def test_worker_recovery_rejects_corrupted_pending_user_input(
    corruption: str,
) -> None:
    async def run() -> None:
        session_id = f"s_corrupt_user_input_{corruption}"
        provider = _ScriptedProvider(
            [("call_1", "ask_user", {"question": "which env?", "options": ["dev"]})]
        )
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool()],
        )
        await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )

        def corrupt_checkpoint(_session, current):
            updated = deepcopy({} if current is None else current)
            pending = updated["pending_user_input"]
            if corruption == "nested-arguments":
                pending["tool_calls"][0]["arguments"] = {"question": "different question"}
            elif corruption == "target-arguments":
                pending["arguments"] = {"question": "different question"}
            else:  # pragma: no cover - parametrization is exhaustive
                raise AssertionError(f"Unknown corruption: {corruption}")
            return updated

        await store.transform_checkpoint(session_id, corrupt_checkpoint)
        await rebind_test_invocation(store, session_id)
        checkpoint_before = await store.load_checkpoint(session_id)
        transcript_before = await store.load_transcript(session_id)
        events_before = await store.load_events(session_id)

        with pytest.raises(
            ValueError,
            match="Pending user-input checkpoint is invalid and cannot be executed",
        ):
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )

        assert await store.load_checkpoint(session_id) == checkpoint_before
        assert await store.load_transcript(session_id) == transcript_before
        assert await store.load_events(session_id) == events_before
        assert len(provider.requests) == 1

    asyncio.run(run())


def test_recover_after_reused_id_prior_round_is_not_wrongly_rejected() -> None:
    # validate_round_recovery_target must scope to the pause's resume window (sweep-sibling of the
    # P1a ledger scoping): a prior round that reused the same tool_call_id (with a terminal event)
    # must NOT make recovery falsely raise "already has a terminal event and does not need recovery".
    store = InMemorySessionStore()
    counting = _CountingTool()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(_TwoRoundProvider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[UserInputTool(), _EchoTool(), counting],
    )
    pause = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s_reuse_rec",
                messages=[Message.text("user", "go")],
            ),
        )
    )
    assert counting.calls == 1  # round 1 ran count(call_1) → a terminal for call_1 exists pre-pause
    input_id = next(e for e in pause if e.type == EventType.SESSION_AWAITING_USER_INPUT).payload[
        "input_id"
    ]
    # Simulate round 2's resolve starting count(call_1) then crashing (started, no terminal in-window).
    asyncio.run(
        store.append_events(
            "s_reuse_rec",
            _crashed_user_input_resume_events(
                asyncio.run(private_events_for_public_events(store, pause)),
                session_id="s_reuse_rec",
                tool_call_id="call_1",
            ),
        )
    )
    stuck = asyncio.run(
        _drain(
            app.resolve_user_input(
                UserInputResponse(session_id="s_reuse_rec", input_id=input_id, answer="a")
            )
        )
    )
    assert stuck[-1].payload.get("manual_recovery_required") is True
    private_terminal = asyncio.run(private_event_for_public_event(store, stuck[-1]))
    assert private_terminal.payload.get("tool_call_id") == "call_1"

    # recover must not be blocked by round 1's stale call_1 terminal event.
    recovered = asyncio.run(
        _drain(
            app.recover_user_input(
                UserInputRecoveryRequest(
                    session_id="s_reuse_rec",
                    input_id=input_id,
                    answer="a",
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="recovered externally",
                )
            )
        )
    )
    assert recovered[-1].type == EventType.SESSION_COMPLETED
    assert counting.calls == 1  # count(call_1) was recovered, never re-executed


def test_recorded_round_outcomes_anchors_from_recovered_interrupted_event() -> None:
    # Direct round identity scopes retry evidence even when the awaiting event was
    # never durably appended after checkpoint publication.
    from cayu.runtime._approval_support import recorded_round_tool_outcomes
    from cayu.runtime.approvals import PendingToolCallApproval

    pending_calls = [PendingToolCallApproval(tool_call_id="call_1", tool_name="count")]
    model_step_id = f"mstep_{'1' * 32}"
    model_attempt_id = f"matt_{'2' * 32}"
    tool_round_id = f"tround_{'3' * 32}"
    identity_payload = {
        "model_step_id": model_step_id,
        "model_attempt_id": model_attempt_id,
        "tool_round_id": tool_round_id,
    }
    events = [
        # A prior round reused call_1 and produced a terminal — must be excluded (before boundary).
        Event(
            type=EventType.TOOL_CALL_COMPLETED,
            session_id="s",
            payload={"tool_call_id": "call_1", "result": ToolResult(content="stale").model_dump()},
        ),
        # No awaiting event (crash before it persisted); recovery finalized the pause here.
        Event(
            type=EventType.SESSION_INTERRUPTED,
            session_id="s",
            payload={"interruption_type": "user_input_required", "user_input": {"input_id": "X"}},
        ),
        # A resume attempt started+completed call_1 before failing to close the transcript.
        Event(
            type=EventType.TOOL_CALL_STARTED,
            session_id="s",
            tool_name="count",
            payload={
                "tool_call_id": "call_1",
                "arguments": {},
                **identity_payload,
            },
        ),
        Event(
            type=EventType.TOOL_CALL_COMPLETED,
            session_id="s",
            tool_name="count",
            payload={
                "tool_call_id": "call_1",
                **identity_payload,
                "result": ToolResult(content="fresh").model_dump(),
            },
        ),
    ]
    recorded = recorded_round_tool_outcomes(
        events=events,
        pending_calls=pending_calls,
        input_id="X",
        tool_round_identity=ToolRoundIdentity(
            model_step_id=model_step_id,
            model_attempt_id=model_attempt_id,
            tool_round_id=tool_round_id,
        ),
    )
    assert "call_1" in recorded  # window is anchored (the awaiting-only code returned {})
    assert recorded["call_1"].result.content == "fresh"  # not the stale prior-round outcome


async def _drain(stream: AsyncIterator[Event]) -> list[Event]:
    return [event async for event in stream]
