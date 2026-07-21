from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

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
    CayuApp,
    ForkSessionRequest,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    NativeStructuredOutputUnsupported,
    ResumeRequest,
    RetryPolicy,
    RunLimits,
    RunRequest,
    Session,
    SessionStatus,
    StructuredOutputSpec,
    StructuredOutputStrategy,
    ToolApprovalRecoveryOutcome,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
    UserInputRecoveryRequest,
    UserInputResponse,
)
from cayu.runtime import _tool_execution as tool_execution
from cayu.tools.user_input import UserInputTool


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
    def __init__(self) -> None:
        super().__init__()
        self.release_calls: dict[str, int] = {}

    async def release_run_fence(self, session_id: str) -> None:
        self.release_calls[session_id] = self.release_calls.get(session_id, 0) + 1
        await super().release_run_fence(session_id)


class _FailingReleaseAfterCleanupStore(_RecordingReleaseStore):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next_release = False

    async def release_run_fence(self, session_id: str) -> None:
        await super().release_run_fence(session_id)
        if self.fail_next_release:
            self.fail_next_release = False
            raise RuntimeError("run fence release unavailable")


class _FailingReleaseBeforeCleanupStore(_RecordingReleaseStore):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next_release = False

    async def release_run_fence(self, session_id: str) -> None:
        self.release_calls[session_id] = self.release_calls.get(session_id, 0) + 1
        if self.fail_next_release:
            self.fail_next_release = False
            raise RuntimeError("run fence release unavailable before cleanup")
        await InMemorySessionStore.release_run_fence(self, session_id)


class _BlockingCommittedRunningTransitionStore(_RecordingReleaseStore):
    def __init__(self) -> None:
        super().__init__()
        self.block_next_running_transition = False
        self.transition_committed: asyncio.Event | None = None
        self.finish_transition: asyncio.Event | None = None

    async def transition_status_and_checkpoint(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        checkpoint_transform,
    ) -> Session:
        session = await super().transition_status_and_checkpoint(
            session_id,
            from_statuses=from_statuses,
            to_status=to_status,
            checkpoint_transform=checkpoint_transform,
        )
        if self.block_next_running_transition and to_status == SessionStatus.RUNNING:
            self.block_next_running_transition = False
            if self.transition_committed is None or self.finish_transition is None:
                raise AssertionError("Transition boundary events were not initialized.")
            self.transition_committed.set()
            await self.finish_transition.wait()
        return session


class _BlockingAbandonedFinalizationStore(_RecordingReleaseStore):
    def __init__(self) -> None:
        super().__init__()
        self.block_next_interrupted_transition = False
        self.finalization_started: asyncio.Event | None = None
        self.finish_finalization: asyncio.Event | None = None

    async def transition_status(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
    ) -> Session:
        if self.block_next_interrupted_transition and to_status == SessionStatus.INTERRUPTED:
            self.block_next_interrupted_transition = False
            if self.finalization_started is None or self.finish_finalization is None:
                raise AssertionError("Finalization boundary events were not initialized.")
            self.finalization_started.set()
            await self.finish_finalization.wait()
        return await super().transition_status(
            session_id,
            from_statuses=from_statuses,
            to_status=to_status,
        )


async def _collect(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]


def _tool_result_parts(transcript) -> list[ToolResultPart]:
    tool_message = next(message for message in transcript if message.role == "tool")
    return [part for part in tool_message.content if isinstance(part, ToolResultPart)]


def _build(first_round, *, tools=None, final_text="done", store=None):
    store = InMemorySessionStore() if store is None else store
    app = CayuApp(session_store=store, enable_logging=False)
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
    assert awaiting.payload["question"] == "Which env?"
    assert awaiting.payload["options"] == ["dev", "prod"]
    assert awaiting.payload["input_id"]
    interrupted = next(e for e in events if e.type == EventType.SESSION_INTERRUPTED)
    assert interrupted.payload["interruption_type"] == "user_input_required"
    assert asyncio.run(store.load("s_pause")).status == SessionStatus.INTERRUPTED
    checkpoint = asyncio.run(store.load_checkpoint("s_pause"))
    assert "pending_user_input" in checkpoint


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
                UserInputResponse(session_id="s_resume", input_id=input_id, answer="prod")
            )
        )
    )

    assert resume_events[-1].type == EventType.SESSION_COMPLETED
    assert {event.environment_name for event in [*pause_events, *resume_events]} == {"optional"}
    session = asyncio.run(store.load("s_resume"))
    assert session is not None
    assert session.environment_name == "optional"
    started = next(
        event
        for event in resume_events
        if event.type == EventType.TOOL_CALL_STARTED
        and event.payload.get("tool_call_id") == "call_1"
    )
    assert started.payload["effect"] == ToolEffect.EXTERNAL.value
    completed = next(
        event
        for event in resume_events
        if event.type == EventType.TOOL_CALL_COMPLETED
        and event.payload.get("tool_call_id") == "call_1"
    )
    assert completed.payload["idempotency_key"] == tool_execution.tool_idempotency_key(
        session_id="s_resume",
        tool_call_id="call_1",
        pause_id=input_id,
    )
    assert asyncio.run(store.load("s_resume")).status == SessionStatus.COMPLETED
    parts = _tool_result_parts(asyncio.run(store.load_transcript("s_resume")))
    ask_part = next(part for part in parts if part.tool_call_id == "call_1")
    assert ask_part.content == "prod"
    assert ask_part.is_error is False
    assert "pending_user_input" not in asyncio.run(store.load_checkpoint("s_resume"))


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
        input_id = next(
            event for event in pause_events if event.type == EventType.SESSION_AWAITING_USER_INPUT
        ).payload["input_id"]

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


def test_resolve_user_input_task_cancellation_finalizes_and_preserves_pending_state() -> None:
    async def run() -> None:
        session_id = "s_resolution_task_cancelled"
        store = _FailingReleaseAfterCleanupStore()
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
        input_id = next(
            event for event in pause_events if event.type == EventType.SESSION_AWAITING_USER_INPUT
        ).payload["input_id"]

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
            assert any(
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
        assert checkpoint["pending_user_input"]["input_id"] == input_id
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
        input_id = next(
            event for event in pause_events if event.type == EventType.SESSION_AWAITING_USER_INPUT
        ).payload["input_id"]

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
        assert checkpoint["pending_user_input"]["input_id"] == input_id
        assert store.release_calls[session_id] - releases_before == 1
        assert app._session_control.has_active_tasks(session_id) is False

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
        input_id = next(
            event for event in pause_events if event.type == EventType.SESSION_AWAITING_USER_INPUT
        ).payload["input_id"]

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
        assert checkpoint["pending_user_input"]["input_id"] == input_id
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
        with pytest.raises(RuntimeError, match="run fence release unavailable before cleanup"):
            await stream.aclose()

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
    answered = next(
        e
        for e in resume_events
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


def test_resolve_user_input_restores_original_run_configuration() -> None:
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


def test_resolve_user_input_explicit_limits_override_persisted_configuration() -> None:
    app, _store, echo = _run_config_app()
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

    events = asyncio.run(
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

    assert "after input" in echo.metadata_by_text
    assert events[-1].type == EventType.SESSION_COMPLETED


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

    resume_events = asyncio.run(
        _drain(
            app.resolve_user_input(
                UserInputResponse(session_id="s_mixed", input_id=input_id, answer="yes")
            )
        )
    )
    assert resume_events[-1].type == EventType.SESSION_COMPLETED
    sibling_events = [
        event
        for event in resume_events
        if event.type in {EventType.TOOL_CALL_STARTED, EventType.TOOL_CALL_COMPLETED}
        and event.payload.get("tool_call_id") in {"call_1", "call_3"}
    ]
    assert sibling_events
    for event in sibling_events:
        call_id = event.payload["tool_call_id"]
        assert event.payload["input_id"] == input_id
        assert event.payload["idempotency_key"] == tool_execution.tool_idempotency_key(
            session_id="s_mixed",
            tool_call_id=call_id,
            pause_id=input_id,
        )
    assert echo.metadata_by_text["first"]["input_id"] == input_id
    assert echo.metadata_by_text["third"]["input_id"] == input_id

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


def test_resolve_user_input_rejects_wrong_input_id() -> None:
    app, _store = _build([("call_1", "ask_user", {"question": "q"})])
    asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant", session_id="s_bad", messages=[Message.text("user", "go")]
            ),
        )
    )
    with pytest.raises(ValueError, match="does not match pending input"):
        asyncio.run(
            _drain(
                app.resolve_user_input(
                    UserInputResponse(session_id="s_bad", input_id="ui_nope", answer="x")
                )
            )
        )


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
    assert awaiting.payload["tool_call_id"] == "call_2"  # paused on the allowed ask_user
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
    input_id = next(
        e for e in pause_events if e.type == EventType.SESSION_AWAITING_USER_INPUT
    ).payload["input_id"]

    resume_events = asyncio.run(
        _drain(
            app.resolve_user_input(
                UserInputResponse(session_id="s_deny", input_id=input_id, answer="ans")
            )
        )
    )
    assert resume_events[-1].type == EventType.SESSION_COMPLETED
    assert any(e.type == EventType.TOOL_CALL_BLOCKED for e in resume_events)
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
    input_id = next(e for e in pause if e.type == EventType.SESSION_AWAITING_USER_INPUT).payload[
        "input_id"
    ]
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


def test_resolve_user_input_rejects_native_structured_output_for_unsupported_provider() -> None:
    # The paused run had no spec, so the resolver's NATIVE spec would be adopted —
    # and must be rejected before the status transition (the fake provider does
    # not support native structured output), leaving the pause resolvable.
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
    input_id = next(e for e in pause if e.type == EventType.SESSION_AWAITING_USER_INPUT).payload[
        "input_id"
    ]

    with pytest.raises(NativeStructuredOutputUnsupported):
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

    session = asyncio.run(store.load("s_native"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


def test_fork_of_paused_session_is_rejected() -> None:
    app, _store = _build([("call_1", "ask_user", {"question": "q"})])
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
    with pytest.raises(RuntimeError, match="awaiting user input cannot be forked"):
        asyncio.run(
            _drain(
                app.fork_session(
                    ForkSessionRequest(source_session_id="s_forksrc", session_id="s_forkchild")
                )
            )
        )


class _FailOnceAppendStore(InMemorySessionStore):
    # Fails the next round-close append once armed (so the initial run — which also uses this
    # method to open the tool round — is unaffected).
    def __init__(self) -> None:
        super().__init__()
        self.armed = False

    async def append_transcript_messages_and_transform_checkpoint(
        self, session_id, messages, checkpoint_transform
    ):
        if self.armed:
            self.armed = False
            raise RuntimeError("simulated append failure")
        return await super().append_transcript_messages_and_transform_checkpoint(
            session_id, messages, checkpoint_transform
        )


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
        store.append_event(
            "s_crash",
            Event(
                type=EventType.TOOL_CALL_STARTED,
                session_id="s_crash",
                agent_name="assistant",
                tool_name="count",
                payload={"tool_call_id": "call_1"},
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
    assert events[-1].payload.get("tool_call_id") == "call_1"
    assert counting.calls == 0  # guard fired before execution — no double-run
    reloaded = asyncio.run(store.load("s_crash"))
    assert reloaded is not None and reloaded.status == SessionStatus.INTERRUPTED


def test_recover_user_input_rejects_native_structured_output_for_unsupported_provider() -> None:
    # The manual-recovery entrance must apply the same pre-transition gate as the
    # other five: an unsupported NATIVE spec raises before the session leaves
    # INTERRUPTED, instead of resuming and failing mid-run via the backstop.
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
    input_id = next(e for e in pause if e.type == EventType.SESSION_AWAITING_USER_INPUT).payload[
        "input_id"
    ]
    asyncio.run(
        store.append_event(
            "s_rec_native",
            Event(
                type=EventType.TOOL_CALL_STARTED,
                session_id="s_rec_native",
                agent_name="assistant",
                tool_name="count",
                payload={"tool_call_id": "call_1"},
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

    with pytest.raises(NativeStructuredOutputUnsupported):
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

    session = asyncio.run(store.load("s_rec_native"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


def test_recover_user_input_supplies_outcome_and_completes() -> None:
    # After a crashed sibling leaves the round on manual_recovery_required, recover_user_input
    # supplies the missing outcome; the round finishes without re-running the sibling, and the
    # re-supplied answer is injected as the ask_user result (it was unrecorded before the crash).
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
                agent_name="assistant", session_id="s_rec", messages=[Message.text("user", "go")]
            ),
        )
    )
    input_id = next(e for e in pause if e.type == EventType.SESSION_AWAITING_USER_INPUT).payload[
        "input_id"
    ]
    # Simulate a prior resume that started `count` but crashed before a terminal event.
    asyncio.run(
        store.append_event(
            "s_rec",
            Event(
                type=EventType.TOOL_CALL_STARTED,
                session_id="s_rec",
                agent_name="assistant",
                tool_name="count",
                payload={"tool_call_id": "call_1"},
            ),
        )
    )
    stuck = asyncio.run(
        _drain(
            app.resolve_user_input(
                UserInputResponse(session_id="s_rec", input_id=input_id, answer="a")
            )
        )
    )
    assert stuck[-1].payload.get("manual_recovery_required") is True

    recovered = asyncio.run(
        _drain(
            app.recover_user_input(
                UserInputRecoveryRequest(
                    session_id="s_rec",
                    input_id=input_id,
                    answer="a",
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="recovered externally",
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
    assert recovered_tool_event.payload["idempotency_key"] == tool_execution.tool_idempotency_key(
        session_id="s_rec",
        tool_call_id="call_1",
        pause_id=input_id,
    )
    assert recovered[-1].type == EventType.SESSION_COMPLETED
    assert counting.calls == 0  # the recovered tool was never re-executed
    checkpoint = asyncio.run(store.load_checkpoint("s_rec"))
    assert checkpoint is not None
    assert "pending_user_input" not in checkpoint
    parts = _tool_result_parts(asyncio.run(store.load_transcript("s_rec")))
    results = {part.tool_call_id: part.content for part in parts}
    assert results["call_1"] == "recovered externally"  # operator-supplied outcome
    assert results["call_2"] == "a"  # ask_user answer injected on continuation


def test_recover_user_input_reconciles_ambiguous_append_acknowledgement() -> None:
    class AmbiguousRecoveryAppendStore(InMemorySessionStore):
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
        await store.append_event(
            session_id,
            Event(
                type=EventType.TOOL_CALL_STARTED,
                session_id=session_id,
                agent_name="assistant",
                tool_name="count",
                payload={"tool_call_id": "call_1"},
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
        await store.append_event(
            session_id,
            Event(
                type=EventType.TOOL_CALL_STARTED,
                session_id=session_id,
                agent_name="assistant",
                tool_name="count",
                payload={"tool_call_id": "call_1"},
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
            assert recovery[-1].id == terminal.id
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
        await store.append_event(
            session_id,
            Event(
                type=EventType.TOOL_CALL_STARTED,
                session_id=session_id,
                agent_name="assistant",
                tool_name="count",
                payload={"tool_call_id": "call_1"},
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
        await store.append_event(
            session_id,
            Event(
                type=EventType.TOOL_CALL_STARTED,
                session_id=session_id,
                agent_name="assistant",
                tool_name="count",
                payload={"tool_call_id": "call_1"},
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
        await store.append_event(
            session_id,
            Event(
                type=EventType.TOOL_CALL_STARTED,
                session_id=session_id,
                agent_name="assistant",
                tool_name="count",
                payload={"tool_call_id": "call_count"},
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
            assert cancellation.args == ("cancel user-input recovery",)
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
    input_id = next(e for e in pause if e.type == EventType.SESSION_AWAITING_USER_INPUT).payload[
        "input_id"
    ]
    # Simulate the crash window: status flipped back to RUNNING with the checkpoint intact.
    asyncio.run(store.update_status("s_crashrec", SessionStatus.RUNNING))
    result = asyncio.run(
        app.recover_incomplete_session(IncompleteSessionRecoveryRequest(session_id="s_crashrec"))
    )
    assert IncompleteSessionRecoveryAction.PENDING_USER_INPUT in result.actions
    assert result.pending_user_input_id == input_id
    interrupted = [e for e in result.events if e.type == EventType.SESSION_INTERRUPTED]
    assert interrupted and interrupted[-1].payload["interruption_type"] == "user_input_required"
    assert interrupted[-1].payload["user_input"]["question"] == "which env?"
    assert asyncio.run(store.load("s_crashrec")).status == SessionStatus.INTERRUPTED


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
        store.append_event(
            "s_reuse_rec",
            Event(
                type=EventType.TOOL_CALL_STARTED,
                session_id="s_reuse_rec",
                agent_name="assistant",
                tool_name="count",
                payload={"tool_call_id": "call_1"},
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
    assert stuck[-1].payload.get("tool_call_id") == "call_1"

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
    # Regression: if the awaiting event was never durably appended (crash after the pending
    # checkpoint), the resume window must anchor from the recovered session.interrupted event so a
    # retry sees the sibling already ran — not re-run it (duplicate side effect).
    from cayu.runtime._approval_support import recorded_round_tool_outcomes
    from cayu.runtime.approvals import PendingToolCallApproval

    pending_calls = [PendingToolCallApproval(tool_call_id="call_1", tool_name="count")]
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
        Event(type=EventType.TOOL_CALL_STARTED, session_id="s", payload={"tool_call_id": "call_1"}),
        Event(
            type=EventType.TOOL_CALL_COMPLETED,
            session_id="s",
            payload={"tool_call_id": "call_1", "result": ToolResult(content="fresh").model_dump()},
        ),
    ]
    recorded = recorded_round_tool_outcomes(
        events=events, pending_calls=pending_calls, input_id="X"
    )
    assert "call_1" in recorded  # window is anchored (the awaiting-only code returned {})
    assert recorded["call_1"].result.content == "fresh"  # not the stale prior-round outcome


async def _drain(stream: AsyncIterator[Event]) -> list[Event]:
    return [event async for event in stream]
