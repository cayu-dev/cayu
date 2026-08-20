from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

import cayu.runtime.execution_profiles as execution_profiles
from cayu.core import AgentSpec, Event, EventType, Message
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
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    RecentTurnsContextPolicy,
    RetryPolicy,
    RunRequest,
    SessionStatus,
    StaticToolExposurePolicy,
    StructuredOutputSpec,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolExposureDecision,
    ToolExposurePolicy,
    ToolExposurePolicyRequest,
    UserInputResponse,
)
from cayu.runtime.hooks import (
    BeforeToolCallHookContext,
    RuntimeHook,
    ToolCallHookContext,
)
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
    await store.transition_status_and_checkpoint(
        session_id,
        from_statuses={session.status},
        to_status=SessionStatus.RUNNING,
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

    pause_events = _run(app, "unexposed-approval-sibling")
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
    [blocked] = [event for event in resume_events if event.type is EventType.TOOL_CALL_BLOCKED]
    assert blocked.tool_name == "hidden"
    assert blocked.payload["reason"] == "not_exposed_in_request"


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

    pause_events = _run(app, "unexposed-input-sibling")
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
            ),
        )
        assert initial_events[-1].type is EventType.SESSION_FAILED
        await _reopen_failed_session_for_recovery(store, session_id)

        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        assert len(provider.requests) == 1
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

    events = _run(app, f"frozen-{failure_kind}")

    assert events[-1].type is EventType.SESSION_COMPLETED
    assert len(policy.requests) == 1
    assert [_tool_names(request) for request in provider.requests] == [
        ["alpha"],
        ["alpha"],
    ]
    assert provider.requests[0].tools == provider.requests[1].tools


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
    assert alpha.calls == [{}]
    assert beta.calls == []
