"""Focused tests for the extracted tool-round execution boundary."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal

import pytest

import cayu.runtime._invocation_secrets as invocation_secrets
from cayu._exception_groups import iter_exception_tree
from cayu.core import AgentSpec, Event, EventType, Message
from cayu.core.events import event_payload_authority_is_runtime_generated
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runners import RunnerExecutionError, attach_cancellation_artifacts
from cayu.runtime import (
    BudgetLimit,
    CayuApp,
    InMemorySessionStore,
    ModelPrice,
    PriceBook,
    RetryPolicy,
    RunLimits,
    RunRequest,
    Session,
    SessionStatus,
)
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import sessions as sessions_module
from cayu.runtime._run_limits import RunLimitGate
from cayu.runtime._session_control import SessionInterruptedByRequest
from cayu.runtime._tool_round_executor import (
    ToolRoundRun,
    _copy_agent_spec,
    _ToolRoundPublicationCoordinator,
)
from cayu.runtime._tool_round_recovery import checkpoint_with_pending_tool_round
from cayu.runtime.execution_profiles import build_execution_profile_identity
from cayu.runtime.execution_units import ToolRoundIdentity
from cayu.runtime.interactions import InteractionStatus, InteractionSummaryEvidence
from cayu.tools._runner import sanitize_runner_failure_group
from cayu.vaults import SecretRedactor


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
        session = await store.transition_status(
            "sess_guard_interrupt",
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
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
        session = await store.transition_status(
            "sess_guard_cancel_interrupt",
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
        )
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
        session = await store.transition_status(
            "sess_grouped_close_failure",
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
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
        session = await store.transition_status(
            "sess_runner_limit",
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
        )
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
        session = await store.transition_status(
            "sess_runner_execute",
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
        )
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
        session = await store.transition_status(
            "sess_tool_round_budget_identity",
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
        )
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
