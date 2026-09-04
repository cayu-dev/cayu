"""Focused tests for the extracted tool-round execution boundary."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest
from tests.core._execution_profile_fixtures import rebind_test_invocation

import cayu.runtime._invocation_secrets as invocation_secrets
from cayu._exception_groups import iter_exception_tree
from cayu.core import AgentSpec, Event, EventType, Message
from cayu.core.events import (
    event_payload_authority_is_runtime_generated,
    event_with_runtime_payload_authority,
)
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runners import RunnerExecutionError, attach_cancellation_artifacts
from cayu.runtime import (
    AfterToolCallDecision,
    BudgetLimit,
    CayuApp,
    InMemorySessionStore,
    ModelPrice,
    PriceBook,
    RetryPolicy,
    RunLimits,
    RunRequest,
    RuntimeHook,
    Session,
    SessionStatus,
    ToolCallHookContext,
)
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _web_access_results as web_access_results
from cayu.runtime import sessions as sessions_module
from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY
from cayu.runtime._run_limits import RunLimitGate
from cayu.runtime._session_control import SessionInterruptedByRequest
from cayu.runtime._tool_round_executor import (
    ToolRoundExecutor,
    ToolRoundRun,
    _copy_agent_spec,
    _durable_payload_utf8_size,
    _project_staged_terminal_event,
    _restore_targeted_tool_invocation_event_authority,
    _ToolRoundPublicationCoordinator,
)
from cayu.runtime._tool_round_recovery import checkpoint_with_pending_tool_round
from cayu.runtime.execution_profiles import build_execution_profile_identity
from cayu.runtime.execution_units import ToolRoundIdentity
from cayu.runtime.interactions import InteractionStatus, InteractionSummaryEvidence
from cayu.runtime.tool_exposure import (
    ResolvedToolExposureAuthority,
    unexposed_tool_result,
)
from cayu.runtime.tool_grants import (
    ResolvedTargetedToolInvocation,
    tool_reference_use_id,
)
from cayu.tools._runner import sanitize_runner_failure_group
from cayu.tools.web import WebFetchTool
from cayu.vaults import SecretRedactor

_CATALOGUE_REVISION = f"sha256:{'c' * 64}"

_TARGETED_AUTHORITY_CONFLICTS = (
    ("dispatch_kind", None),
    ("dispatch_kind", "native"),
    ("model_tool_name", "remember"),
    ("grant_id", f"sha256:{'a' * 64}"),
    ("use_id", f"sha256:{'b' * 64}"),
    ("effective_tool_id", "cayu:other"),
    ("catalogue_revision", f"sha256:{'d' * 64}"),
    ("descriptor_version", f"sha256:{'e' * 64}"),
    ("schema_fingerprint", f"sha256:{'f' * 64}"),
    ("arguments_sha256", f"sha256:{'7' * 64}"),
    ("invocation_id", "other-invocation"),
)


def _resolved_gateway_tool_call() -> runtime_records.ToolCallRequest:
    grant_id = f"sha256:{'1' * 64}"
    arguments_sha256 = f"sha256:{'6' * 64}"
    use_id = tool_reference_use_id(
        grant_id=grant_id,
        session_id="session-targeted-authority",
        interaction_id="interaction-targeted-authority",
        model_step_id="model-step-targeted-authority",
        outer_tool_call_id="call-targeted-authority",
        arguments_sha256=arguments_sha256,
        invocation_id="invocation-targeted-authority",
    )
    invocation = ResolvedTargetedToolInvocation(
        dispatch_kind="gateway",
        model_tool_name="call_tool",
        tool_ref="cayu_authority_v1.test-targeted-authority",
        grant_id=grant_id,
        use_id=use_id,
        session_id="session-targeted-authority",
        interaction_id="interaction-targeted-authority",
        tool_id="cayu:remember",
        effective_tool_name="remember",
        catalogue_revision=f"sha256:{'3' * 64}",
        descriptor_version=f"sha256:{'4' * 64}",
        schema_fingerprint=f"sha256:{'5' * 64}",
        model_step_id="model-step-targeted-authority",
        outer_tool_call_id="call-targeted-authority",
        arguments_sha256=arguments_sha256,
        invocation_id="invocation-targeted-authority",
    )
    return runtime_records.ToolCallRequest(
        id=invocation.outer_tool_call_id,
        name=invocation.effective_tool_name,
        arguments={"fact": "remember"},
        targeted_tool_grant_id=grant_id,
        model_tool_name=invocation.model_tool_name,
        targeted_tool_invocation=invocation,
    )


@pytest.mark.parametrize(
    ("field_name", "conflicting_value"),
    _TARGETED_AUTHORITY_CONFLICTS,
)
def test_restored_targeted_terminal_rejects_every_retained_authority_conflict(
    field_name: str,
    conflicting_value: object,
) -> None:
    tool_call = _resolved_gateway_tool_call()
    terminal = Event(
        type=EventType.TOOL_CALL_FAILED,
        session_id="session-targeted-authority",
        tool_name=tool_call.name,
        payload={
            "tool_call_id": tool_call.id,
            field_name: conflicting_value,
            "result": {"content": "failed", "is_error": True},
        },
    )

    with pytest.raises(
        RuntimeError,
        match="conflicts with its durable invocation authority",
    ):
        _restore_targeted_tool_invocation_event_authority(terminal, tool_call)


@pytest.mark.parametrize(
    ("event_tool_name", "event_tool_call_id"),
    (("other", "call-targeted-authority"), ("remember", "other-call")),
)
def test_restored_targeted_terminal_rejects_resolved_invocation_identity_conflict(
    event_tool_name: str,
    event_tool_call_id: str,
) -> None:
    tool_call = _resolved_gateway_tool_call()
    terminal = Event(
        type=EventType.TOOL_CALL_FAILED,
        session_id="session-targeted-authority",
        tool_name=event_tool_name,
        payload={
            "tool_call_id": event_tool_call_id,
            "result": {"content": "failed", "is_error": True},
        },
    )

    with pytest.raises(RuntimeError, match="conflicts with its resolved invocation"):
        _restore_targeted_tool_invocation_event_authority(terminal, tool_call)


def test_restored_targeted_terminal_accepts_only_attested_private_projection() -> None:
    tool_call = _resolved_gateway_tool_call()
    terminal = Event(
        type=EventType.TOOL_CALL_FAILED,
        session_id="session-targeted-authority",
        tool_name=tool_call.name,
        payload={
            "tool_call_id": tool_call.id,
            "grant_id": tool_call.targeted_tool_invocation.grant_id,
            "result": {"content": "failed", "is_error": True},
        },
    )
    attested = event_with_runtime_payload_authority(terminal, "grant_id")
    projected = attested.model_copy(
        update={
            "payload": {
                **attested.payload,
                "grant_id": PRIVATE_EVENT_AUTHORITY,
            }
        }
    )

    restored = _restore_targeted_tool_invocation_event_authority(projected, tool_call)

    assert restored.payload["grant_id"] == tool_call.targeted_tool_invocation.grant_id


class _NoExecutableToolAgent:
    def executable_tool(self, _tool_name: str) -> None:
        return None


async def _stage_targeted_terminal_from_publication_entrance(
    terminal: Event,
    tool_call: runtime_records.ToolCallRequest,
) -> list[Event]:
    executor = object.__new__(ToolRoundExecutor)
    executor._secret_redactor = SecretRedactor()
    staged: list[Event] = []

    async def stage(event: Event, *_args: object) -> None:
        staged.append(event)

    async for _ in executor.emit_tool_call_result_with_hooks(
        event=terminal,
        session=SimpleNamespace(id=terminal.session_id),
        registered_agent=_NoExecutableToolAgent(),
        registered_environment=None,
        tool_call=tool_call,
        result=ToolResult(content="failed", is_error=True),
        task_id=None,
        deferred_terminal_stager=stage,
        publication_snapshot=object(),
    ):
        raise AssertionError("Deferred terminal staging unexpectedly yielded an event.")
    return staged


@pytest.mark.parametrize(
    ("field_name", "conflicting_value"),
    _TARGETED_AUTHORITY_CONFLICTS,
)
def test_tool_result_publication_rejects_retained_authority_before_staging(
    field_name: str,
    conflicting_value: object,
) -> None:
    tool_call = _resolved_gateway_tool_call()
    terminal = Event(
        type=EventType.TOOL_CALL_FAILED,
        session_id="session-targeted-authority",
        tool_name=tool_call.name,
        payload={
            "tool_call_id": tool_call.id,
            field_name: conflicting_value,
            "result": {"content": "failed", "is_error": True},
        },
    )

    async def run() -> None:
        with pytest.raises(
            RuntimeError,
            match="conflicts with its durable invocation authority",
        ):
            await _stage_targeted_terminal_from_publication_entrance(terminal, tool_call)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("event_tool_name", "event_tool_call_id"),
    (("other", "call-targeted-authority"), ("remember", "other-call")),
)
def test_tool_result_publication_rejects_resolved_identity_before_staging(
    event_tool_name: str,
    event_tool_call_id: str,
) -> None:
    tool_call = _resolved_gateway_tool_call()
    terminal = Event(
        type=EventType.TOOL_CALL_FAILED,
        session_id="session-targeted-authority",
        tool_name=event_tool_name,
        payload={
            "tool_call_id": event_tool_call_id,
            "result": {"content": "failed", "is_error": True},
        },
    )

    async def run() -> None:
        with pytest.raises(RuntimeError, match="conflicts with its resolved invocation"):
            await _stage_targeted_terminal_from_publication_entrance(terminal, tool_call)

    asyncio.run(run())


@pytest.mark.parametrize("retained_projection", ("absent", "attested_private"))
def test_tool_result_publication_restores_only_safe_authority_before_staging(
    retained_projection: str,
) -> None:
    tool_call = _resolved_gateway_tool_call()
    terminal = Event(
        type=EventType.TOOL_CALL_FAILED,
        session_id="session-targeted-authority",
        tool_name=tool_call.name,
        payload={
            "tool_call_id": tool_call.id,
            "result": {"content": "failed", "is_error": True},
        },
    )
    if retained_projection == "attested_private":
        invocation = tool_call.targeted_tool_invocation
        assert invocation is not None
        terminal = event_with_runtime_payload_authority(
            terminal.model_copy(
                update={
                    "payload": {
                        **terminal.payload,
                        "grant_id": invocation.grant_id,
                    }
                }
            ),
            "grant_id",
        ).model_copy(
            update={
                "payload": {
                    **terminal.payload,
                    "grant_id": PRIVATE_EVENT_AUTHORITY,
                }
            }
        )

    [staged] = asyncio.run(_stage_targeted_terminal_from_publication_entrance(terminal, tool_call))
    expected = tool_call.targeted_tool_invocation
    assert expected is not None
    assert staged.payload["grant_id"] == expected.grant_id
    assert event_payload_authority_is_runtime_generated(
        staged,
        field_name="grant_id",
        value=expected.grant_id,
    )


class _FakeProvider(ModelProvider):
    name = "fake"

    def __init__(self, events: list[ModelStreamEvent]) -> None:
        self.events = events

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        for event in self.events:
            yield event


class _SideEffectTool(Tool):
    spec = ToolSpec(
        name="side_effect",
        description="Record execution.",
        input_schema={"type": "object", "properties": {}},
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls.append(args)
        return ToolResult(content="recorded")


class _OversizedBoundedResultTool(Tool):
    spec = ToolSpec(
        name="side_effect",
        description="Return a result larger than the declared durable contract.",
        input_schema={"type": "object", "properties": {}},
        max_terminal_payload_bytes=64 * 1024,
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx, args
        self.calls += 1
        return ToolResult(content="x" * (70 * 1024))


class _BoundedSmallResultTool(Tool):
    spec = ToolSpec(
        name="side_effect",
        description="Return a small result under a bounded durable contract.",
        input_schema={
            "type": "object",
            "properties": {"blob": {"type": "string"}},
            "additionalProperties": False,
        },
        max_terminal_payload_bytes=64 * 1024,
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx
        self.calls.append(args)
        return ToolResult(content="ok")


class _BoundedWorkspaceResultTool(_BoundedSmallResultTool):
    spec = ToolSpec(
        name="side_effect",
        description="Return a bounded result after a workspace-owned effect.",
        input_schema={
            "type": "object",
            "properties": {"blob": {"type": "string"}},
            "additionalProperties": False,
        },
        parallel_safe=False,
        workspace_mutation=True,
        max_terminal_payload_bytes=64 * 1024,
    )


def _app_with_completed_session(
    session_id: str,
    *,
    total_tokens: int = 11,
) -> tuple[CayuApp, InMemorySessionStore, _SideEffectTool]:
    store = InMemorySessionStore()
    provider = _FakeProvider(
        [
            ModelStreamEvent.text_delta("final answer"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": total_tokens - 4,
                        "output_tokens": 4,
                        "total_tokens": total_tokens,
                    },
                }
            ),
        ]
    )
    tool = _SideEffectTool()
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    async def run() -> None:
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "answer")],
            )
        ):
            pass

    asyncio.run(run())
    return app, store, tool


def _limit_gate(
    app: CayuApp,
    session,
    *,
    limits: RunLimits,
    budget_limits: tuple[BudgetLimit, ...] = (),
) -> RunLimitGate:
    return RunLimitGate(
        app._run_limit_controller,
        session=session,
        agent_name="assistant",
        environment_name=None,
        limits=limits,
        budget_limits=budget_limits,
        run_started_at=time.monotonic(),
        run_baseline=None,
        budget_baseline_events=[],
        budget_notify_events=[],
    )


async def _tool_round_run(
    app: CayuApp,
    session: Session,
    *,
    limits: RunLimits,
    budget_limits: tuple[BudgetLimit, ...] = (),
) -> ToolRoundRun:
    interaction_id = f"interaction-{session.id}"
    started_at = datetime.now(UTC)
    start_event_id = f"{session.id}:interaction-started"
    await app.session_store.append_event(
        session.id,
        Event(
            id=start_event_id,
            type=EventType.INTERACTION_STARTED,
            session_id=session.id,
            interaction_id=interaction_id,
            timestamp=started_at,
            agent_name="assistant",
            payload=InteractionSummaryEvidence(
                status=InteractionStatus.ACTIVE,
                start_event_id=start_event_id,
                started_at=started_at,
            ).model_dump(mode="json"),
        ),
    )
    sessions_module._activate_session_interaction(
        session.id,
        interaction_id,
    )
    return app._tool_round_executor.create_run(
        session=session,
        registered_agent=app._get_registered_agent("assistant"),
        registered_environment=None,
        environment_name=None,
        limit_gate=_limit_gate(
            app,
            session,
            limits=limits,
            budget_limits=budget_limits,
        ),
        request_metadata={},
        task_id=None,
        structured_output=None,
        thinking=None,
        max_steps=16,
        limits=RunLimits(),
        budget_limits=budget_limits,
        retry_policy=RetryPolicy(),
        run_started_at=time.monotonic(),
        turn_usage_tracker=None,
        active_run=None,
    )


def _tool_call(call_id: str = "call_1") -> runtime_records.ToolCallRequest:
    return runtime_records.ToolCallRequest(id=call_id, name="side_effect", arguments={})


def _tool_round_identity() -> ToolRoundIdentity:
    return ToolRoundIdentity(
        model_step_id=f"mstep_{'1' * 32}",
        model_attempt_id=f"matt_{'2' * 32}",
        tool_round_id=f"tround_{'3' * 32}",
    )


def test_staged_terminal_profile_authority_is_owned_by_the_active_round() -> None:
    identity = _tool_round_identity()
    profile = build_execution_profile_identity(
        runtime_name="cayu",
        runtime_version="test",
        provider_name="fake",
        model="fake-model",
        durable_system_prompt="",
        direct_tools=[],
        tool_catalogue_revision=f"sha256:{'c' * 64}",
    )
    coordinator = _ToolRoundPublicationCoordinator(
        session_id="session-staged-profile-authority",
        tool_round_identity=identity,
        session_store=InMemorySessionStore(),
        redactor=SecretRedactor(),
        execution_profile=profile,
    )
    terminal = Event(
        type=EventType.TOOL_CALL_FAILED,
        session_id="session-staged-profile-authority",
        payload={
            **identity.payload(),
            "tool_call_id": "call_1",
            "tool_name": "side_effect",
            "result": ToolResult(content="interrupted", is_error=True).model_dump(),
        },
    )

    restored = coordinator.restore_staged_event_authority(terminal)

    assert restored.payload["execution_profile_fingerprint"] == profile.fingerprint
    assert event_payload_authority_is_runtime_generated(
        restored,
        field_name="execution_profile_fingerprint",
        value=profile.fingerprint,
    )
    with pytest.raises(
        RuntimeError,
        match="conflicts with its execution profile owner",
    ):
        coordinator.restore_staged_event_authority(
            terminal.model_copy(
                update={
                    "payload": {
                        **terminal.payload,
                        "execution_profile_fingerprint": "0" * 64,
                    }
                }
            )
        )


def test_staged_terminal_exposure_authority_is_owned_by_the_frozen_snapshot() -> None:
    identity = _tool_round_identity()
    exposure = ResolvedToolExposureAuthority(
        profile_id="tool-free",
        catalogue_revision=_CATALOGUE_REVISION,
        tool_names=(),
        registered_count=1,
        ceiling_count=1,
        fingerprint="a" * 64,
    )
    coordinator = _ToolRoundPublicationCoordinator(
        session_id="session-staged-exposure-authority",
        tool_round_identity=identity,
        session_store=InMemorySessionStore(),
        redactor=SecretRedactor(),
        execution_profile=None,
        tool_exposure=exposure,
    )
    terminal = Event(
        type=EventType.TOOL_CALL_BLOCKED,
        session_id="session-staged-exposure-authority",
        tool_name="side_effect",
        payload={
            **identity.payload(),
            "tool_call_id": "call_1",
            "blocked_by": "tool_exposure",
            "reason": "not_exposed_in_request",
            "profile_id": exposure.profile_id,
            "exposure_fingerprint": exposure.fingerprint,
            "arguments_state": "unavailable",
            "result": unexposed_tool_result().model_dump(mode="json"),
        },
    )

    restored = coordinator.restore_staged_event_authority(terminal)

    for field_name, expected in (
        ("profile_id", exposure.profile_id),
        ("exposure_fingerprint", exposure.fingerprint),
    ):
        assert restored.payload[field_name] == expected
        assert event_payload_authority_is_runtime_generated(
            restored,
            field_name=field_name,
            value=expected,
        )

    with pytest.raises(
        ValueError,
        match="conflicts with its frozen exposure authority",
    ):
        coordinator.restore_staged_event_authority(
            terminal.model_copy(
                update={
                    "payload": {
                        **terminal.payload,
                        "exposure_fingerprint": "b" * 64,
                    }
                }
            )
        )

    unowned_coordinator = _ToolRoundPublicationCoordinator(
        session_id="session-staged-exposure-authority",
        tool_round_identity=identity,
        session_store=InMemorySessionStore(),
        redactor=SecretRedactor(),
        execution_profile=None,
    )
    with pytest.raises(RuntimeError, match="has no durable exposure owner"):
        unowned_coordinator.restore_staged_event_authority(terminal)


def test_staged_web_access_authority_survives_only_owned_durable_reconstruction() -> None:
    identity = _tool_round_identity()
    result = ToolResult(
        content="Access is blocked.",
        structured={
            "access": {
                "schema_version": 1,
                "outcome": "bot_challenge",
                "source": "http_response",
                "signal": "status_code",
                "destination_fingerprint": "a" * 64,
                "status_code": 401,
                "retry_after_seconds": None,
                "retry_after_unrepresentable": False,
            }
        },
        is_error=True,
    )
    terminal = web_access_results.attest_runtime_web_access_result(
        Event(
            type=EventType.TOOL_CALL_FAILED,
            session_id="session-staged-web-access-authority",
            tool_name="web_fetch",
            payload={
                **identity.payload(),
                "tool_call_id": "call_1",
                "result": result.model_dump(mode="json"),
            },
        ),
        result,
        tool=WebFetchTool(),
    )
    marker = terminal.payload[web_access_results.WEB_ACCESS_RESULT_AUTHORITY_FIELD]
    persisted = Event.model_validate(terminal.model_dump(mode="json"))
    redactor = SecretRedactor("bot_challenge")

    projected = _project_staged_terminal_event(
        persisted,
        redactor=redactor,
        trust_persisted_tool_result_authority=True,
    )
    projected_access = projected.payload["result"]["structured"]["access"]
    assert projected_access["outcome"] == "bot_challenge"
    assert projected.payload[web_access_results.WEB_ACCESS_RESULT_AUTHORITY_FIELD] == marker

    untrusted = _project_staged_terminal_event(persisted, redactor=redactor)
    assert web_access_results.WEB_ACCESS_RESULT_AUTHORITY_FIELD not in untrusted.payload
    assert untrusted.payload["result"]["structured"]["access"]["outcome"] == ("[REDACTED_SECRET]")

    coordinator = _ToolRoundPublicationCoordinator(
        session_id="session-staged-web-access-authority",
        tool_round_identity=identity,
        session_store=InMemorySessionStore(),
        redactor=redactor,
        execution_profile=None,
    )
    restored = coordinator.restore_staged_event_authority(persisted)
    assert event_payload_authority_is_runtime_generated(
        restored,
        field_name=web_access_results.WEB_ACCESS_RESULT_AUTHORITY_FIELD,
        value=marker,
    )

    tampered_payload = persisted.model_dump(mode="json")["payload"]
    tampered_payload["result"]["structured"]["access"]["outcome"] = "consent_required"
    with pytest.raises(ValueError, match="authority is malformed"):
        _project_staged_terminal_event(
            persisted.model_copy(update={"payload": tampered_payload}),
            redactor=redactor,
            trust_persisted_tool_result_authority=True,
        )


def test_tool_round_agent_copy_rejects_agent_spec_subclasses() -> None:
    class _DerivedAgentSpec(AgentSpec):
        pass

    with pytest.raises(TypeError, match="Agent registration requires an AgentSpec"):
        _copy_agent_spec(_DerivedAgentSpec(name="assistant", model="fake-model"))


def test_tool_round_interrupt_close_ignores_unrequested_cancellation():
    app, store, _ = _app_with_completed_session("sess_guard_cancel")

    async def scenario() -> tuple[list[Event], list[Message]]:
        session = await store.load("sess_guard_cancel")
        assert session is not None
        runner = await _tool_round_run(app, session, limits=RunLimits())
        messages: list[Message] = []
        events = [
            event
            async for event in runner.close_after_interrupt(
                asyncio.CancelledError(),
                messages=messages,
                tool_calls=[_tool_call()],
                tool_outcomes=[],
                tool_round_identity=_tool_round_identity(),
            )
        ]
        return events, messages

    events, messages = asyncio.run(scenario())

    assert events == []
    assert messages == []
    transcript = asyncio.run(store.load_transcript("sess_guard_cancel"))
    assert [message.role for message in transcript] == ["user", "assistant"]


def test_tool_round_interrupt_close_persists_missing_results():
    app, store, _ = _app_with_completed_session("sess_guard_interrupt")

    async def scenario() -> tuple[list[Event], list[Message]]:
        session = await rebind_test_invocation(store, "sess_guard_interrupt")
        runner = await _tool_round_run(app, session, limits=RunLimits())
        tool_calls = [_tool_call()]
        checkpoint, _pending_round = checkpoint_with_pending_tool_round(
            await store.load_checkpoint(session.id),
            agent_name="assistant",
            environment_name=None,
            task_id=None,
            tool_calls=tool_calls,
            policy_outcomes=None,
            structured_output=None,
            tool_round_identity=_tool_round_identity(),
        )
        await store.checkpoint(session.id, checkpoint)
        messages = await store.load_transcript(session.id)
        events = [
            event
            async for event in runner.close_after_interrupt(
                SessionInterruptedByRequest(session.id),
                messages=messages,
                tool_calls=tool_calls,
                tool_outcomes=[],
                tool_round_identity=_tool_round_identity(),
            )
        ]
        return events, messages

    events, messages = asyncio.run(scenario())

    assert [event.type for event in events] == [EventType.TOOL_CALL_FAILED]
    assert events[0].payload["tool_call_id"] == "call_1"
    assert events[0].payload["tool_round_id"] == f"tround_{'3' * 32}"
    assert messages[-1].role == "tool"
    transcript = asyncio.run(store.load_transcript("sess_guard_interrupt"))
    assert [message.role for message in transcript] == ["user", "assistant", "tool"]


def test_tool_round_interrupt_close_handles_requested_cancellation():
    app, store, _ = _app_with_completed_session("sess_guard_cancel_interrupt")

    async def scenario() -> tuple[list[Event], list[Message]]:
        session = await rebind_test_invocation(store, "sess_guard_cancel_interrupt")
        session = await store.update_status(
            session.id,
            SessionStatus.INTERRUPTING,
        )
        runner = await _tool_round_run(app, session, limits=RunLimits())
        tool_calls = [_tool_call()]
        checkpoint, _pending_round = checkpoint_with_pending_tool_round(
            await store.load_checkpoint(session.id),
            agent_name="assistant",
            environment_name=None,
            task_id=None,
            tool_calls=tool_calls,
            policy_outcomes=None,
            structured_output=None,
            tool_round_identity=_tool_round_identity(),
        )
        await store.checkpoint(session.id, checkpoint)
        messages = await store.load_transcript(session.id)
        events = [
            event
            async for event in runner.close_after_interrupt(
                asyncio.CancelledError(),
                messages=messages,
                tool_calls=tool_calls,
                tool_outcomes=[],
                tool_round_identity=_tool_round_identity(),
            )
        ]
        return events, messages

    events, messages = asyncio.run(scenario())

    assert [event.type for event in events] == [EventType.TOOL_CALL_FAILED]
    assert events[0].payload["tool_call_id"] == "call_1"
    assert messages[-1].role == "tool"


def test_grouped_cancellation_remains_authoritative_when_interrupt_closure_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, store, _ = _app_with_completed_session("sess_grouped_close_failure")
    closure_secret = "interrupt-closure-secret-canary-ABCDEFGHIJKLMNOP"

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        session = await rebind_test_invocation(store, "sess_grouped_close_failure")
        runner = await _tool_round_run(app, session, limits=RunLimits())
        tool_calls = [_tool_call()]
        checkpoint, _pending_round = checkpoint_with_pending_tool_round(
            await store.load_checkpoint(session.id),
            agent_name="assistant",
            environment_name=None,
            task_id=None,
            tool_calls=tool_calls,
            policy_outcomes=None,
            structured_output=None,
            tool_round_identity=_tool_round_identity(),
        )
        await store.checkpoint(session.id, checkpoint)
        started = asyncio.Event()

        async def grouped_call_stream(**kwargs):
            del kwargs
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:
                await store.update_status(session.id, SessionStatus.INTERRUPTING)
                attach_cancellation_artifacts(
                    cancellation,
                    [
                        {
                            "type": "cayu.runner_cleanup.v1",
                            "adapter": "microsandbox",
                            "action": "kill_command",
                            "status": "failed",
                            "timeout_s": 5.0,
                        }
                    ],
                )
                group = sanitize_runner_failure_group(
                    BaseExceptionGroup(
                        "runner cleanup failed",
                        [cancellation, RuntimeError("runner cleanup failed")],
                    ),
                    caller_cancelled=True,
                )
                for candidate in iter_exception_tree(group):
                    if not isinstance(candidate, asyncio.CancelledError):
                        continue
                    invocation_secrets.initialize_cancellation_evidence(candidate)
                    invocation_secrets.set_cancellation_tool_call_id(
                        candidate,
                        tool_calls[0].id,
                    )
                    invocation_secrets.set_cancellation_redactor(
                        candidate,
                        SecretRedactor(),
                    )
                raise group from None
            if False:  # pragma: no cover - declares this as an async iterator
                yield

        async def fail_close(request):
            del request
            if False:  # pragma: no cover - declares this as an async iterator
                yield
            raise RuntimeError(closure_secret)

        monkeypatch.setattr(
            runner,
            "_run_tool_calls_sequential",
            grouped_call_stream,
        )
        monkeypatch.setattr(
            runner._executor,
            "_close_interrupted_round",
            fail_close,
        )

        async def consume_round() -> None:
            async for _ in runner.run(
                messages=await store.load_transcript(session.id),
                tool_calls=tool_calls,
                tool_round_identity=_tool_round_identity(),
            ):
                pass

        task = asyncio.create_task(consume_round())
        await started.wait()
        task.cancel("operator cancellation")
        cancelling = task.cancelling()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value, cancelling, task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())
    assert cancelling == 1
    assert cancelled is True
    assert isinstance(cancellation.__cause__, BaseExceptionGroup)
    failures = [
        candidate
        for candidate in iter_exception_tree(cancellation.__cause__)
        if not isinstance(candidate, BaseExceptionGroup)
    ]
    assert any(isinstance(failure, RunnerExecutionError) for failure in failures)
    assert any(
        type(failure) is RuntimeError and str(failure) == "Interrupted tool-round closure failed."
        for failure in failures
    )
    assert closure_secret not in repr(cancellation)
    assert closure_secret not in repr(cancellation.__cause__)


def test_tool_round_interrupt_close_rejects_unrelated_exceptions():
    app, store, _ = _app_with_completed_session("sess_guard_type_error")

    async def scenario() -> None:
        session = await store.load("sess_guard_type_error")
        assert session is not None
        runner = await _tool_round_run(app, session, limits=RunLimits())
        async for _ in runner.close_after_interrupt(
            ValueError("not an interrupt"),
            messages=[],
            tool_calls=[],
            tool_outcomes=[],
            tool_round_identity=_tool_round_identity(),
        ):
            pass

    with pytest.raises(TypeError, match="Unsupported interrupt exception"):
        asyncio.run(scenario())


def test_tool_round_runner_stops_for_limit_before_tool_side_effects():
    app, store, tool = _app_with_completed_session("sess_runner_limit")

    async def scenario() -> tuple[list[Event], bool]:
        session = await rebind_test_invocation(store, "sess_runner_limit")
        runner = await _tool_round_run(
            app,
            session,
            limits=RunLimits(max_total_tokens=10),
        )
        tool_calls = [_tool_call()]
        checkpoint, _pending_round = checkpoint_with_pending_tool_round(
            await store.load_checkpoint(session.id),
            agent_name="assistant",
            environment_name=None,
            task_id=None,
            tool_calls=tool_calls,
            policy_outcomes=None,
            structured_output=None,
            tool_round_identity=_tool_round_identity(),
        )
        await store.checkpoint(session.id, checkpoint)
        events = [
            event
            async for event in runner.run(
                messages=await store.load_transcript(session.id),
                tool_calls=tool_calls,
                tool_round_identity=_tool_round_identity(),
            )
        ]
        return events, runner.stopped_for_limit

    events, stopped_for_limit = asyncio.run(scenario())

    assert stopped_for_limit is True
    assert [event.type for event in events] == [
        EventType.SESSION_LIMIT_REACHED,
        EventType.TOOL_CALL_FAILED,
        EventType.INTERACTION_INTERRUPTED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert events[1].payload["reason"] == "limit_reached"
    assert tool.calls == []


def test_tool_round_runner_executes_tool_round_and_persists_results():
    app, store, tool = _app_with_completed_session("sess_runner_execute")

    async def scenario() -> tuple[list[Event], list[Message], bool]:
        session = await rebind_test_invocation(store, "sess_runner_execute")
        runner = await _tool_round_run(
            app,
            session,
            limits=RunLimits(max_total_tokens=100),
        )
        tool_calls = [_tool_call()]
        source_checkpoint = await store.load_checkpoint(session.id)
        checkpoint, pending_round = checkpoint_with_pending_tool_round(
            source_checkpoint,
            agent_name="assistant",
            environment_name=None,
            task_id=None,
            tool_calls=tool_calls,
            policy_outcomes=None,
            structured_output=None,
            tool_round_identity=_tool_round_identity(),
        )
        # A live pre-versioning checkpoint is upgraded when the newly evaluated
        # policy plan is published; it must not fail as unrecoverable legacy state.
        assert pending_round.policy_context_version is None
        await store.checkpoint(session.id, checkpoint)
        messages = await store.load_transcript(session.id)
        events = [
            event
            async for event in runner.run(
                messages=messages,
                tool_calls=tool_calls,
                tool_round_identity=_tool_round_identity(),
            )
        ]
        await store.release_run_fence(session.id)
        return events, messages, runner.stopped_for_limit

    events, messages, stopped_for_limit = asyncio.run(scenario())

    assert stopped_for_limit is False
    assert [event.type for event in events] == [
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
    ]
    assert tool.calls == [{}]
    assert messages[-1].role == "tool"
    transcript = asyncio.run(store.load_transcript("sess_runner_execute"))
    assert transcript[-1].role == "tool"


def test_tool_result_over_declared_terminal_limit_is_bounded_effect_authority() -> None:
    store = InMemorySessionStore()
    provider = _FakeProvider(
        [
            ModelStreamEvent.text_delta("initial"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    tool = _OversizedBoundedResultTool()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    async def scenario() -> Event:
        session_id = "sess_declared_terminal_overflow"
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "initialize")],
            )
        ):
            pass
        session = await rebind_test_invocation(store, session_id)
        runner = await _tool_round_run(app, session, limits=RunLimits())
        tool_calls = [_tool_call()]
        checkpoint, _pending = checkpoint_with_pending_tool_round(
            await store.load_checkpoint(session.id),
            agent_name="assistant",
            environment_name=None,
            task_id=None,
            tool_calls=tool_calls,
            policy_outcomes=None,
            structured_output=None,
            tool_round_identity=_tool_round_identity(),
        )
        await store.checkpoint(session.id, checkpoint)
        events = [
            event
            async for event in runner.run(
                messages=await store.load_transcript(session.id),
                tool_calls=tool_calls,
                tool_round_identity=_tool_round_identity(),
            )
        ]
        return next(event for event in events if event.type is EventType.TOOL_CALL_FAILED)

    terminal = asyncio.run(scenario())

    assert tool.calls == 1
    assert terminal.payload["terminal_outcome"] == "invalid_tool_output"
    assert terminal.payload["tool_effect"] == "external"
    assert terminal.payload["outcome_unknown"] is True
    assert terminal.payload["manual_reconciliation_required"] is True
    assert terminal.payload["result"]["is_error"] is True
    assert len(terminal.payload["result"]["content"].encode("utf-8")) < 1024


def test_large_arguments_do_not_consume_the_declared_result_payload_limit() -> None:
    store = InMemorySessionStore()
    provider = _FakeProvider(
        [
            ModelStreamEvent.text_delta("initial"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    tool = _BoundedWorkspaceResultTool()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    async def scenario() -> Event:
        session_id = "sess_large_arguments_small_terminal_result"
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "initialize")],
            )
        ):
            pass
        session = await rebind_test_invocation(store, session_id)
        runner = await _tool_round_run(app, session, limits=RunLimits())
        tool_call = runtime_records.ToolCallRequest(
            id="call_1",
            name="side_effect",
            arguments={"blob": "x" * 70_000},
        )
        checkpoint, _pending = checkpoint_with_pending_tool_round(
            await store.load_checkpoint(session.id),
            agent_name="assistant",
            environment_name=None,
            task_id=None,
            tool_calls=[tool_call],
            policy_outcomes=None,
            structured_output=None,
            tool_round_identity=_tool_round_identity(),
        )
        await store.checkpoint(session.id, checkpoint)
        events = [
            event
            async for event in runner.run(
                messages=await store.load_transcript(session.id),
                tool_calls=[tool_call],
                tool_round_identity=_tool_round_identity(),
            )
        ]
        return next(
            event
            for event in events
            if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
        )

    terminal = asyncio.run(scenario())

    assert tool.calls == [{"blob": "x" * 70_000}]
    assert terminal.type is EventType.TOOL_CALL_COMPLETED
    assert terminal.payload["result"]["content"] == "ok"
    assert "terminal_outcome" not in terminal.payload
    publication = app.tool_terminal_publication_status()
    assert publication.maximum_reserved_round_bytes == (
        128 * 1024 + _durable_payload_utf8_size({"blob": "x" * 70_000})
    )
    assert publication.oversized_offloads >= 1
    assert publication.active_round_reservations == 0


def test_after_hook_result_rewrite_is_rebounded_before_staged_publication() -> None:
    class OversizedRewriteHook(RuntimeHook):
        async def after_tool_call(
            self,
            context: ToolCallHookContext,
        ) -> AfterToolCallDecision:
            del context
            return AfterToolCallDecision(
                action="modify",
                modified_result=ToolResult(content="y" * 300_000),
            )

    store = InMemorySessionStore()
    provider = _FakeProvider(
        [
            ModelStreamEvent.text_delta("initial"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    tool = _BoundedWorkspaceResultTool()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        runtime_hooks=[OversizedRewriteHook()],
    )

    async def scenario() -> Event:
        session_id = "sess_hook_rewrite_terminal_overflow"
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "initialize")],
            )
        ):
            pass
        session = await rebind_test_invocation(store, session_id)
        runner = await _tool_round_run(app, session, limits=RunLimits())
        tool_calls = [_tool_call()]
        checkpoint, _pending = checkpoint_with_pending_tool_round(
            await store.load_checkpoint(session.id),
            agent_name="assistant",
            environment_name=None,
            task_id=None,
            tool_calls=tool_calls,
            policy_outcomes=None,
            structured_output=None,
            tool_round_identity=_tool_round_identity(),
        )
        await store.checkpoint(session.id, checkpoint)
        events = [
            event
            async for event in runner.run(
                messages=await store.load_transcript(session.id),
                tool_calls=tool_calls,
                tool_round_identity=_tool_round_identity(),
            )
        ]
        return next(event for event in events if event.type is EventType.TOOL_CALL_FAILED)

    terminal = asyncio.run(scenario())
    publication = app.tool_terminal_publication_status()

    assert tool.calls == [{}]
    assert terminal.payload["terminal_outcome"] == "invalid_tool_output"
    assert terminal.payload["result"]["is_error"] is True
    assert len(terminal.payload["result"]["content"].encode("utf-8")) < 1024
    assert publication.maximum_staged_bytes == _durable_payload_utf8_size(terminal.payload)
    assert publication.maximum_staged_bytes <= publication.maximum_reserved_round_bytes
    assert publication.active_round_reservations == 0


def test_tool_round_budget_gate_retains_the_originating_model_attempt() -> None:
    app, store, _ = _app_with_completed_session("sess_tool_round_budget_identity")
    budget_limit = BudgetLimit(
        scope="session",
        max_estimated_cost=Decimal("1"),
        pricing=PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="fake",
                    model="fake-model",
                    input_per_million=Decimal("1000000"),
                    output_per_million=Decimal("1000000"),
                ),
            )
        ),
        action="notify",
    )
    identity = _tool_round_identity()

    async def scenario() -> list[Event]:
        session = await rebind_test_invocation(store, "sess_tool_round_budget_identity")
        runner = await _tool_round_run(
            app,
            session,
            limits=RunLimits(),
            budget_limits=(budget_limit,),
        )
        tool_calls = [_tool_call()]
        checkpoint, _pending_round = checkpoint_with_pending_tool_round(
            await store.load_checkpoint(session.id),
            agent_name="assistant",
            environment_name=None,
            task_id=None,
            tool_calls=tool_calls,
            policy_outcomes=None,
            structured_output=None,
            tool_round_identity=identity,
        )
        await store.checkpoint(session.id, checkpoint)
        events = [
            event
            async for event in runner.run(
                messages=await store.load_transcript(session.id),
                tool_calls=tool_calls,
                tool_round_identity=identity,
            )
        ]
        await store.release_run_fence(session.id)
        return events

    events = asyncio.run(scenario())

    reached = next(event for event in events if event.type == EventType.BUDGET_LIMIT_REACHED)
    assert reached.payload["model_step_id"] == identity.model_step_id
    assert reached.payload["model_attempt_id"] == identity.model_attempt_id
