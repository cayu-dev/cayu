"""Recovery cancellation contracts."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import pytest
from tests.core._execution_profile_fixtures import rebind_test_invocation
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_recovery_support import (
    FailingTerminalToolEventStore,
    _crashed_tool_round_app,
)
from tests.runtime.runtime_session_support import (
    _private_record_for_public_event,
    _runtime_exception_graph,
    assert_only_model_step_publication_checkpoint,
    collect_events,
    collect_interrupt_events,
    collect_tool_round_recovery_events,
)
from tests.runtime.runtime_tool_support import PureRecordingTool, SideEffectTool

import cayu.runtime._session_control as session_control_module
import cayu.runtime.execution_profiles as execution_profiles_module
import cayu.sessions._model_completion_publication as model_completion_publication_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import ToolApprovalRecoveryOutcome
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import _recovery_coordinator as recovery_coordinator_module
from cayu.runtime import _recovery_ownership as recovery_ownership_module
from cayu.sessions import _invocation_scope as invocation_scope
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.records import Session, SessionStatus
from cayu.sessions.requests import InterruptSessionRequest, RunRequest
from cayu.tools.rounds import ToolRoundRecoveryRequest


def test_cayu_app_recover_tool_round_task_cancellation_finalizes_session():
    class BlockingRecoveryProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.continuation_started = asyncio.Event()
            self.never_complete = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                yield ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={},
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            self.continuation_started.set()
            await self.never_complete.wait()
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    async def scenario() -> None:
        session_id = "sess_tool_round_manual_cancelled"
        store = FailingTerminalToolEventStore()
        provider = BlockingRecoveryProvider()
        tool = PureRecordingTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

        initial_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
            ),
        )
        assert initial_events[-1].type == EventType.SESSION_FAILED
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        round_id = checkpoint["pending_tool_round"]["tool_round_id"]

        recovery_task = asyncio.create_task(
            collect_tool_round_recovery_events(
                app,
                ToolRoundRecoveryRequest(
                    session_id=session_id,
                    round_id=round_id,
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="side effect verified externally",
                ),
            )
        )
        await asyncio.wait_for(provider.continuation_started.wait(), timeout=10)
        assert recovery_task.cancelling() == 0
        recovery_task.cancel("cancel manual tool recovery")
        assert recovery_task.cancelling() == 1
        with pytest.raises(asyncio.CancelledError) as cancellation:
            await recovery_task
        assert cancellation.value.args == ("cancel manual tool recovery",)
        assert recovery_task.cancelled()
        assert recovery_task.cancelling() == 1

        session = await store.load(session_id)
        assert session is not None
        assert session.status == SessionStatus.INTERRUPTED
        checkpoint = await store.load_checkpoint(session_id)
        assert_only_model_step_publication_checkpoint(checkpoint)
        events = await store.load_events(session_id)
        assert events[-1].type == EventType.SESSION_INTERRUPTED
        assert events[-1].payload["abandoned"] is True
        assert tool.calls == [{}]
        assert app._session_control.has_active_tasks(session_id) is False

    asyncio.run(scenario())


@pytest.mark.parametrize("local_marker", ["request", "signal"])
def test_manual_recovery_watcher_defers_to_local_active_interruption(
    monkeypatch: pytest.MonkeyPatch, local_marker: str
) -> None:
    session_id = "sess_manual_recovery_local_interrupt_owner"
    app, store, _tool, _checkpoint = _crashed_tool_round_app(session_id)

    async def scenario() -> None:
        loaded = await store.load(session_id)
        assert loaded is not None
        await store.transition_status(
            session_id, from_statuses={loaded.status}, to_status=SessionStatus.INTERRUPTING
        )
        owner_release = asyncio.Event()
        owner = asyncio.create_task(owner_release.wait())
        app._session_control.register_active_task(
            session_id, owner, task_id=None, task_started=True, task_finished=False
        )
        if local_marker == "request":
            app._session_control.begin_interruption_request(session_id)
        else:
            app._session_control.signal_interrupt(session_id)
        inspected = asyncio.Event()
        original_require_session = app._recovery_ownership.require_session

        async def observed_require_session(candidate_id):
            session = await original_require_session(candidate_id)
            inspected.set()
            return session

        monkeypatch.setattr(app._recovery_ownership, "require_session", observed_require_session)
        stop = asyncio.Event()
        watcher = asyncio.create_task(
            app._recovery_ownership.watch_manual_recovery_interruption(
                session_id=session_id, interrupted_baseline_id=None, stop=stop
            )
        )
        try:
            await asyncio.wait_for(inspected.wait(), timeout=10)
            assert not watcher.done()
            stop.set()
            assert await asyncio.wait_for(watcher, timeout=10) is False
        finally:
            stop.set()
            owner_release.set()
            await asyncio.gather(watcher, owner, return_exceptions=True)
            app._session_control.unregister_active_task(session_id, owner)
            app._session_control.end_interruption_request(session_id)
            app._session_control.discard_interrupt_signal(session_id)

    asyncio.run(scenario())


@pytest.mark.parametrize("deferred_response", [False, True])
def test_cayu_app_recover_tool_round_operator_interrupts_blocked_continuation(
    monkeypatch: pytest.MonkeyPatch, deferred_response: bool
) -> None:
    active_wait_attempts = session_control_module.ACTIVE_INTERRUPTED_EVENT_WAIT_ATTEMPTS
    if deferred_response:
        monkeypatch.setattr(session_control_module, "ACTIVE_INTERRUPTED_EVENT_WAIT_ATTEMPTS", 0)

    class BlockingRecoveryProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.continuation_started = asyncio.Event()
            self.continuation_cancelled = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                yield ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={},
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            self.continuation_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.continuation_cancelled.set()
                raise

    async def scenario() -> None:
        session_id = "sess_tool_round_manual_operator_interrupt"
        store = FailingTerminalToolEventStore()
        provider = BlockingRecoveryProvider()
        tool = PureRecordingTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

        finalization_entered = asyncio.Event()
        release_finalization = asyncio.Event()
        watcher_rechecked = asyncio.Event()
        finalization_cancelled = asyncio.Event()
        original_load = app._session_engine.session_store.load
        original_require = app._recovery_ownership.require_session
        watcher_checks = 0

        async def hold_terminal_publication(candidate_id):
            if (
                deferred_response
                and candidate_id == session_id
                and app._session_control.is_emitting_interrupted(session_id)
                and not finalization_entered.is_set()
            ):
                finalization_entered.set()
                try:
                    await release_finalization.wait()
                except asyncio.CancelledError:
                    finalization_cancelled.set()
                    raise
            return await original_load(candidate_id)

        async def observe_watcher(candidate_id):
            nonlocal watcher_checks
            current = await original_require(candidate_id)
            if finalization_entered.is_set() and not release_finalization.is_set():
                watcher_checks += 1
                if watcher_checks >= 2:
                    watcher_rechecked.set()
            return current

        monkeypatch.setattr(app._session_engine.session_store, "load", hold_terminal_publication)
        monkeypatch.setattr(app._recovery_ownership, "require_session", observe_watcher)

        initial_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
            ),
        )
        assert initial_events[-1].type == EventType.SESSION_FAILED
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        recovery_task = asyncio.create_task(
            collect_tool_round_recovery_events(
                app,
                ToolRoundRecoveryRequest(
                    session_id=session_id,
                    round_id=checkpoint["pending_tool_round"]["tool_round_id"],
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="side effect verified externally",
                ),
            )
        )
        await asyncio.wait_for(provider.continuation_started.wait(), timeout=10)

        interrupt_request = InterruptSessionRequest(
            session_id=session_id,
            reason="operator stopped manual recovery",
            metadata={"source": "blocked-continuation-regression"},
        )
        interruption_events = None
        interrupt_task = asyncio.create_task(collect_interrupt_events(app, interrupt_request))
        try:
            if deferred_response:
                try:
                    await asyncio.wait_for(finalization_entered.wait(), timeout=10)
                    # The run has unregistered, but its terminal publisher still
                    # owns cleanup. Let the durable watcher inspect it twice.
                    await asyncio.wait_for(watcher_rechecked.wait(), timeout=5)
                    assert not finalization_cancelled.is_set()
                    assert not recovery_task.done()
                finally:
                    release_finalization.set()
            interruption_events = await asyncio.wait_for(interrupt_task, timeout=15)
        except TimeoutError as error:
            # The request can return while the active run is still finalizing.
            # Completion and replay below must still prove one exact terminal.
            assert str(error) == f"Session interruption is still finalizing: {session_id}"
        finally:
            monkeypatch.setattr(
                session_control_module,
                "ACTIVE_INTERRUPTED_EVENT_WAIT_ATTEMPTS",
                active_wait_attempts,
            )
            if not interrupt_task.done():
                interrupt_task.cancel()
            await asyncio.gather(interrupt_task, return_exceptions=True)
        recovery_events = await asyncio.wait_for(recovery_task, timeout=15)
        assert recovery_task.cancelling() == 0
        assert not recovery_task.cancelled()
        if interruption_events is None:
            interruption_events = await asyncio.wait_for(
                collect_interrupt_events(app, interrupt_request), timeout=15
            )

        assert provider.continuation_cancelled.is_set()
        assert interruption_events[-1].type == EventType.SESSION_INTERRUPTED
        assert recovery_events[-1].id == interruption_events[-1].id
        assert interruption_events[-1].payload["interruption_type"] == "operator_requested"
        assert interruption_events[-1].payload["reason"] == "operator stopped manual recovery"
        assert interruption_events[-1].payload["metadata"] == {
            "source": "blocked-continuation-regression"
        }
        session = await store.load(session_id)
        assert session is not None and session.status == SessionStatus.INTERRUPTED
        terminal_interruptions = [
            event
            for event in await store.load_events(session_id)
            if event.type == EventType.SESSION_INTERRUPTED
        ]
        assert len(terminal_interruptions) == 1
        records = await store.query_events(EventQuery(session_id=session_id, limit=100))
        assert (
            _private_record_for_public_event(records, interruption_events[-1]).event.id
            == terminal_interruptions[0].id
        )
        assert await app.drain_background_interruptions(timeout_s=1) is True
        final_checkpoint = await store.load_checkpoint(session_id)
        assert final_checkpoint is not None
        assert "pending_session_interrupt" not in final_checkpoint
        assert "pending_tool_round" not in final_checkpoint
        assert "pending_operator_interruption_cascade" not in final_checkpoint
        assert (
            model_completion_publication_module.model_step_publication_from_checkpoint(
                final_checkpoint
            )
            is not None
        )
        assert tool.calls == [{}]
        assert app._session_control.has_active_tasks(session_id) is False

    asyncio.run(scenario())


def test_cayu_app_recover_tool_round_heartbeat_loss_stops_continuation_before_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_order: list[str] = []

    class OrderedReleaseStore(FailingTerminalToolEventStore):
        invocation_lifecycle_command_version = 1

        async def release_session_invocation(self, command):
            cleanup_order.append("run_fence_release")
            return await super().release_session_invocation(command)

    class BlockingRecoveryProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.continuation_started = asyncio.Event()
            self.continuation_cancelled = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                yield ModelStreamEvent.tool_call(
                    id="call_1",
                    name="side_effect",
                    arguments={},
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            self.continuation_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup_order.append("provider_cancelled")
                self.continuation_cancelled.set()
                raise

    async def scenario() -> None:
        session_id = "sess_tool_round_manual_heartbeat_lost"
        store = OrderedReleaseStore()
        provider = BlockingRecoveryProvider()
        tool = PureRecordingTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

        initial_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
            ),
        )
        assert initial_events[-1].type == EventType.SESSION_FAILED
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        round_id = checkpoint["pending_tool_round"]["tool_round_id"]
        cleanup_order.clear()

        async def fail_heartbeat(**_kwargs) -> None:
            await provider.continuation_started.wait()
            cleanup_order.append("heartbeat_lost")
            raise RuntimeError("manual recovery claim heartbeat lost")

        async def record_environment_abort(
            *,
            session_id: str,
            original_error,
            execution_profile=None,
            invocation_context=None,
        ) -> None:
            del execution_profile, invocation_context
            assert isinstance(original_error, RuntimeError)
            checkpoint_during_abort = await store.load_checkpoint(session_id)
            assert checkpoint_during_abort is not None
            assert "incomplete_session_recovery_claim" in checkpoint_during_abort
            cleanup_order.append("environment_abort")

        original_finalize = app._session_finalization.finalize_abandoned_session_by_id

        async def record_abandoned_finalization(
            session_id: str,
            *,
            propagate_interaction_publication_rejection: bool = False,
            registered_agent=None,
            registered_environment=None,
            execution_profile=None,
            invocation_context=None,
        ) -> None:
            cleanup_order.append("abandoned_finalization")
            await original_finalize(
                session_id,
                propagate_interaction_publication_rejection=(
                    propagate_interaction_publication_rejection
                ),
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
            )

        monkeypatch.setattr(
            app._recovery_ownership,
            "heartbeat",
            fail_heartbeat,
        )
        monkeypatch.setattr(
            app._environment_lifecycle,
            "abort_environment_setup",
            record_environment_abort,
        )
        monkeypatch.setattr(
            app._session_finalization,
            "finalize_abandoned_session_by_id",
            record_abandoned_finalization,
        )

        with pytest.raises(
            RuntimeError,
            match="manual recovery claim heartbeat lost",
        ):
            await asyncio.wait_for(
                collect_tool_round_recovery_events(
                    app,
                    ToolRoundRecoveryRequest(
                        session_id=session_id,
                        round_id=round_id,
                        tool_call_id="call_1",
                        outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                        message="side effect verified externally",
                    ),
                ),
                timeout=5,
            )

        assert provider.continuation_cancelled.is_set()
        assert cleanup_order.count("abandoned_finalization") == 1
        assert cleanup_order.count("environment_abort") == 1
        assert cleanup_order.index("provider_cancelled") < cleanup_order.index(
            "abandoned_finalization"
        )
        assert cleanup_order.index("abandoned_finalization") < cleanup_order.index(
            "environment_abort"
        )
        assert cleanup_order.index("environment_abort") < cleanup_order.index("run_fence_release")
        session = await store.load(session_id)
        assert session is not None and session.status == SessionStatus.INTERRUPTED
        checkpoint = await store.load_checkpoint(session_id)
        assert_only_model_step_publication_checkpoint(checkpoint)
        assert tool.calls == [{}]
        assert app._session_control.has_active_tasks(session_id) is False

    asyncio.run(scenario())


def test_cayu_app_recover_tool_round_heartbeat_loss_cleans_up_while_consumer_is_paused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_order: list[str] = []

    class ObservableReleaseStore(FailingTerminalToolEventStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.observe_release = False
            self.release_finished = asyncio.Event()

        async def release_session_invocation(self, command):
            if self.observe_release:
                cleanup_order.append("run_fence_release")
            result = await super().release_session_invocation(command)
            if self.observe_release:
                self.release_finished.set()
            return result

    session_id = "sess_tool_round_manual_heartbeat_lost_while_yielded"
    store = ObservableReleaseStore()
    app, store, tool, checkpoint = _crashed_tool_round_app(
        session_id, store=store, tool=PureRecordingTool()
    )
    request = ToolRoundRecoveryRequest(
        session_id=session_id,
        round_id=checkpoint["pending_tool_round"]["tool_round_id"],
        tool_call_id="call_1",
        outcome=ToolApprovalRecoveryOutcome.COMPLETED,
        message="side effect verified externally",
    )

    async def scenario() -> None:
        allow_heartbeat_failure = asyncio.Event()

        async def fail_heartbeat(**_kwargs) -> None:
            await allow_heartbeat_failure.wait()
            cleanup_order.append("heartbeat_lost")
            raise RuntimeError("manual recovery claim heartbeat lost while yielding")

        async def record_environment_abort(
            *,
            session_id: str,
            original_error,
            execution_profile=None,
            invocation_context=None,
        ) -> None:
            del execution_profile, invocation_context
            assert isinstance(original_error, RuntimeError)
            checkpoint_during_abort = await store.load_checkpoint(session_id)
            assert checkpoint_during_abort is not None
            assert "incomplete_session_recovery_claim" in checkpoint_during_abort
            cleanup_order.append("environment_abort")

        original_finalize = app._session_finalization.finalize_abandoned_session_by_id

        async def record_abandoned_finalization(
            session_id: str,
            *,
            propagate_interaction_publication_rejection: bool = False,
            registered_agent=None,
            registered_environment=None,
            execution_profile=None,
            invocation_context=None,
        ) -> None:
            cleanup_order.append("abandoned_finalization")
            await original_finalize(
                session_id,
                propagate_interaction_publication_rejection=(
                    propagate_interaction_publication_rejection
                ),
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
            )

        monkeypatch.setattr(
            app._recovery_ownership,
            "heartbeat",
            fail_heartbeat,
        )
        monkeypatch.setattr(
            app._environment_lifecycle,
            "abort_environment_setup",
            record_environment_abort,
        )
        monkeypatch.setattr(
            app._session_finalization,
            "finalize_abandoned_session_by_id",
            record_abandoned_finalization,
        )
        store.observe_release = True

        stream = app.recover_tool_round(request)
        first_event = await asyncio.wait_for(anext(stream), timeout=5)
        assert first_event.type == EventType.SESSION_RESUMED
        allow_heartbeat_failure.set()

        # Do not request another event yet. The background supervisor must still
        # observe lease loss and finish owned cleanup while delivery is paused.
        await asyncio.wait_for(store.release_finished.wait(), timeout=5)
        # Invocation release precedes presence shutdown and claim removal.
        # Observe the supervisor's complete cleanup, not that intermediate write.
        assert await app.drain_recovery_cleanups(timeout_s=5)
        interrupted = await store.load(session_id)
        assert interrupted is not None and interrupted.status == SessionStatus.INTERRUPTED
        checkpoint_after_cleanup = await store.load_checkpoint(session_id)
        assert checkpoint_after_cleanup is not None
        assert checkpoint_after_cleanup["pending_tool_round"]["tool_round_id"] == request.round_id
        assert "incomplete_session_recovery_claim" not in checkpoint_after_cleanup
        assert cleanup_order[0] == "heartbeat_lost"
        assert cleanup_order.count("abandoned_finalization") >= 1
        assert cleanup_order.count("environment_abort") == 1
        assert cleanup_order.count("run_fence_release") == 1
        environment_abort_index = cleanup_order.index("environment_abort")
        assert all(
            index < environment_abort_index
            for index, operation in enumerate(cleanup_order)
            if operation == "abandoned_finalization"
        )
        assert environment_abort_index < cleanup_order.index("run_fence_release")

        with pytest.raises(
            RuntimeError,
            match="manual recovery claim heartbeat lost while yielding",
        ):
            await asyncio.wait_for(anext(stream), timeout=5)
        assert tool.calls == [{}]
        assert app._session_control.has_active_tasks(session_id) is False

    asyncio.run(scenario())


def test_cayu_app_recover_tool_round_remains_interruptible_while_consumer_is_paused() -> None:
    session_id = "sess_tool_round_manual_interrupt_while_yielded"
    app, store, tool, checkpoint = _crashed_tool_round_app(session_id, tool=PureRecordingTool())
    request = ToolRoundRecoveryRequest(
        session_id=session_id,
        round_id=checkpoint["pending_tool_round"]["tool_round_id"],
        tool_call_id="call_1",
        outcome=ToolApprovalRecoveryOutcome.COMPLETED,
        message="side effect verified externally",
    )

    async def scenario() -> None:
        stream = app.recover_tool_round(request)
        one_event_consumer = asyncio.create_task(anext(stream))
        first_event = await asyncio.wait_for(one_event_consumer, timeout=5)
        assert first_event.type == EventType.SESSION_RESUMED
        assert one_event_consumer.done()
        assert app._session_control.active_runs(session_id) == ()
        assert app._session_control.has_active_tasks(session_id)

        interruption_events = await asyncio.wait_for(
            collect_interrupt_events(
                app,
                InterruptSessionRequest(
                    session_id=session_id,
                    reason="operator stopped paused manual recovery",
                    metadata={"source": "paused-delivery-regression"},
                ),
            ),
            timeout=5,
        )
        assert interruption_events[-1].type == EventType.SESSION_INTERRUPTED
        assert interruption_events[-1].payload["interruption_type"] == "operator_requested"
        assert interruption_events[-1].payload["reason"] == (
            "operator stopped paused manual recovery"
        )
        assert interruption_events[-1].payload["metadata"] == {
            "source": "paused-delivery-regression"
        }

        await stream.aclose()
        interrupted = await store.load(session_id)
        assert interrupted is not None and interrupted.status == SessionStatus.INTERRUPTED
        checkpoint_after_interrupt = await store.load_checkpoint(session_id)
        assert checkpoint_after_interrupt is not None
        assert checkpoint_after_interrupt["pending_tool_round"]["tool_round_id"] == request.round_id
        assert "incomplete_session_recovery_claim" not in checkpoint_after_interrupt
        terminal_interruptions = [
            event
            for event in await store.load_events(session_id)
            if event.type == EventType.SESSION_INTERRUPTED
        ]
        assert len(terminal_interruptions) == 1
        assert terminal_interruptions[0].payload["reason"] == (
            "operator stopped paused manual recovery"
        )
        assert terminal_interruptions[0].payload.get("abandoned") is not True
        assert tool.calls == [{}]
        assert app._session_control.has_active_tasks(session_id) is False

    asyncio.run(scenario())


def test_cayu_app_recover_tool_round_observes_cross_worker_interrupt_while_paused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recovery_ownership_module,
        "_MANUAL_RECOVERY_INTERRUPT_POLL_INTERVAL_SECONDS",
        0.01,
    )
    session_id = "sess_tool_round_cross_worker_interrupt_while_yielded"
    app, store, tool, checkpoint = _crashed_tool_round_app(session_id, tool=PureRecordingTool())
    other_worker = CayuApp(session_store=store, enable_logging=False)
    other_worker.register_provider(FakeProvider([]), default=True)
    other_worker.register_agent(AgentSpec(name="assistant", model="fake-model"))
    request = ToolRoundRecoveryRequest(
        session_id=session_id,
        round_id=checkpoint["pending_tool_round"]["tool_round_id"],
        tool_call_id="call_1",
        outcome=ToolApprovalRecoveryOutcome.COMPLETED,
        message="side effect verified externally",
    )

    async def scenario() -> None:
        stream = app.recover_tool_round(request)
        interrupt_task: asyncio.Task[list[Event]] | None = None
        try:
            first_event = await asyncio.wait_for(anext(stream), timeout=5)
            assert first_event.type == EventType.SESSION_RESUMED
            assert app._session_control.has_active_tasks(session_id)

            interrupt_task = asyncio.create_task(
                collect_interrupt_events(
                    other_worker,
                    InterruptSessionRequest(
                        session_id=session_id,
                        reason="operator stopped recovery from another worker",
                        metadata={"source": "cross-worker-paused-delivery-regression"},
                    ),
                )
            )
            interruption_events = await asyncio.wait_for(interrupt_task, timeout=5)
            assert interruption_events[-1].type == EventType.SESSION_INTERRUPTED
            assert interruption_events[-1].payload["interruption_type"] == ("operator_requested")
            assert interruption_events[-1].payload["reason"] == (
                "operator stopped recovery from another worker"
            )
            assert interruption_events[-1].payload["metadata"] == {
                "source": "cross-worker-paused-delivery-regression"
            }
        finally:
            await stream.aclose()
            if interrupt_task is not None and not interrupt_task.done():
                interrupt_task.cancel()
                await asyncio.gather(interrupt_task, return_exceptions=True)

        interrupted = await store.load(session_id)
        assert interrupted is not None and interrupted.status == SessionStatus.INTERRUPTED
        checkpoint_after_interrupt = await store.load_checkpoint(session_id)
        assert checkpoint_after_interrupt is not None
        assert checkpoint_after_interrupt["pending_tool_round"]["tool_round_id"] == request.round_id
        assert "incomplete_session_recovery_claim" not in checkpoint_after_interrupt
        terminal_interruptions = [
            event
            for event in await store.load_events(session_id)
            if event.type == EventType.SESSION_INTERRUPTED
        ]
        assert len(terminal_interruptions) == 1
        assert terminal_interruptions[0].payload["reason"] == (
            "operator stopped recovery from another worker"
        )
        assert terminal_interruptions[0].payload.get("abandoned") is not True
        assert tool.calls == [{}]
        assert app._session_control.has_active_tasks(session_id) is False
        assert other_worker._session_control.has_active_tasks(session_id) is False

    asyncio.run(scenario())


def test_cayu_app_recover_tool_round_preserves_cross_worker_interrupt_during_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailAfterClaimStore(FailingTerminalToolEventStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.arm_recovery_load_failure = False
            self.recovery_load_started = asyncio.Event()
            self.allow_recovery_load_failure = asyncio.Event()
            self.recovery_loader: asyncio.Task[Any] | None = None

        async def load_events(self, session_id: str) -> list[Event]:
            current_task = asyncio.current_task()
            if self.arm_recovery_load_failure and self.recovery_loader is None:
                assert current_task is not None
                self.recovery_loader = current_task
                self.recovery_load_started.set()
                await self.allow_recovery_load_failure.wait()
                self.arm_recovery_load_failure = False
                raise RuntimeError("pre-persistence recovery read failed")
            return await super().load_events(session_id)

    monkeypatch.setattr(
        recovery_ownership_module,
        "_MANUAL_RECOVERY_INTERRUPT_POLL_INTERVAL_SECONDS",
        60,
    )
    session_id = "sess_tool_round_cross_worker_interrupt_during_restore"
    store = FailAfterClaimStore()
    app, store, tool, checkpoint = _crashed_tool_round_app(
        session_id, store=store, tool=PureRecordingTool()
    )
    other_worker = CayuApp(session_store=store, enable_logging=False)
    other_worker.register_provider(FakeProvider([]), default=True)
    other_worker.register_agent(AgentSpec(name="assistant", model="fake-model"))
    request = ToolRoundRecoveryRequest(
        session_id=session_id,
        round_id=checkpoint["pending_tool_round"]["tool_round_id"],
        tool_call_id="call_1",
        outcome=ToolApprovalRecoveryOutcome.COMPLETED,
        message="side effect verified externally",
    )

    async def scenario() -> None:
        store.arm_recovery_load_failure = True
        recovery_task = asyncio.create_task(collect_tool_round_recovery_events(app, request))
        interrupt_task: asyncio.Task[list[Event]] | None = None
        try:
            await asyncio.wait_for(store.recovery_load_started.wait(), timeout=10)
            running = await store.load(session_id)
            assert running is not None and running.status == SessionStatus.RUNNING

            interrupt_task = asyncio.create_task(
                collect_interrupt_events(
                    other_worker,
                    InterruptSessionRequest(
                        session_id=session_id,
                        reason="operator won the failed recovery race",
                        metadata={"source": "restore-race-regression"},
                    ),
                )
            )
            for _ in range(500):
                interrupting = await store.load(session_id)
                if interrupting is not None and interrupting.status == SessionStatus.INTERRUPTING:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("operator interruption did not become durable")
            store.allow_recovery_load_failure.set()
            interruption_events, recovery_events = await asyncio.gather(
                asyncio.wait_for(interrupt_task, timeout=5),
                asyncio.wait_for(recovery_task, timeout=5),
            )
            assert interruption_events[-1].type == EventType.SESSION_INTERRUPTED
        finally:
            store.allow_recovery_load_failure.set()
            for task in (recovery_task, interrupt_task):
                if task is not None and not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

        assert recovery_events[-1].id == interruption_events[-1].id
        assert recovery_events[-1].payload["interruption_type"] == "operator_requested"
        assert recovery_events[-1].payload["reason"] == ("operator won the failed recovery race")
        assert recovery_events[-1].payload["metadata"] == {"source": "restore-race-regression"}
        interrupted = await store.load(session_id)
        assert interrupted is not None and interrupted.status == SessionStatus.INTERRUPTED
        checkpoint_after_interrupt = await store.load_checkpoint(session_id)
        assert checkpoint_after_interrupt is not None
        assert checkpoint_after_interrupt["pending_tool_round"]["tool_round_id"] == request.round_id
        assert "pending_session_interrupt" not in checkpoint_after_interrupt
        assert "incomplete_session_recovery_claim" not in checkpoint_after_interrupt
        terminal_interruptions = [
            event
            for event in await store.load_events(session_id)
            if event.type == EventType.SESSION_INTERRUPTED
        ]
        assert len(terminal_interruptions) == 1
        assert terminal_interruptions[0].payload.get("manual_recovery_required") is not True
        assert tool.calls == [{}]
        assert app._session_control.has_active_tasks(session_id) is False
        assert other_worker._session_control.has_active_tasks(session_id) is False

    asyncio.run(scenario())


def test_cayu_app_recover_tool_round_aclose_reports_cleanup_failure() -> None:
    class FailingRecoveryReleaseStore(FailingTerminalToolEventStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.fail_release = False

        async def release_session_invocation(self, command):
            if self.fail_release:
                self.fail_release = False
                raise RuntimeError("manual recovery run fence release unavailable")
            return await super().release_session_invocation(command)

    session_id = "sess_tool_round_manual_aclose_cleanup_failure"
    store = FailingRecoveryReleaseStore()
    app, store, tool, checkpoint = _crashed_tool_round_app(
        session_id, store=store, tool=PureRecordingTool()
    )
    request = ToolRoundRecoveryRequest(
        session_id=session_id,
        round_id=checkpoint["pending_tool_round"]["tool_round_id"],
        tool_call_id="call_1",
        outcome=ToolApprovalRecoveryOutcome.COMPLETED,
        message="side effect verified externally",
    )

    async def scenario() -> None:
        stream = app.recover_tool_round(request)
        first_event = await asyncio.wait_for(anext(stream), timeout=5)
        assert first_event.type == EventType.SESSION_RESUMED

        store.fail_release = True
        with pytest.raises(
            RuntimeError,
            match="manual recovery run fence release unavailable",
        ):
            await stream.aclose()

        interrupted = await store.load(session_id)
        assert interrupted is not None and interrupted.status == SessionStatus.INTERRUPTED
        checkpoint_after_close = await store.load_checkpoint(session_id)
        assert checkpoint_after_close is not None
        assert "incomplete_session_recovery_claim" not in checkpoint_after_close
        assert checkpoint_after_close["pending_tool_round"]["tool_round_id"] == request.round_id
        assert tool.calls == [{}]
        assert app._session_control.has_active_tasks(session_id) is False

    asyncio.run(scenario())


def test_cayu_app_recover_tool_round_athrow_finalizes_before_return() -> None:
    session_id = "sess_tool_round_manual_athrow_cleanup"
    app, store, tool, checkpoint = _crashed_tool_round_app(session_id, tool=PureRecordingTool())
    request = ToolRoundRecoveryRequest(
        session_id=session_id,
        round_id=checkpoint["pending_tool_round"]["tool_round_id"],
        tool_call_id="call_1",
        outcome=ToolApprovalRecoveryOutcome.COMPLETED,
        message="side effect completed externally",
    )

    async def scenario() -> None:
        releases_before = store.release_calls
        stream = app.recover_tool_round(request)
        first_event = await asyncio.wait_for(anext(stream), timeout=5)
        assert first_event.type == EventType.SESSION_RESUMED

        claimed_checkpoint = await store.load_checkpoint(session_id)
        assert claimed_checkpoint is not None
        assert "incomplete_session_recovery_claim" in claimed_checkpoint
        assert app._session_control.has_active_tasks(session_id) is True
        # Ownership belongs to the delegated stream, not the consumer task.
        assert invocation_scope._current_session_run_epoch(session_id) is None

        injected = RuntimeError("consumer rejected recovery event")
        with pytest.raises(RuntimeError, match="consumer rejected recovery event") as exc_info:
            await stream.athrow(injected)
        assert exc_info.value is injected

        interrupted = await store.load(session_id)
        assert interrupted is not None and interrupted.status == SessionStatus.INTERRUPTED
        checkpoint_after_throw = await store.load_checkpoint(session_id)
        assert checkpoint_after_throw is not None
        assert checkpoint_after_throw["pending_tool_round"]["tool_round_id"] == request.round_id
        assert "incomplete_session_recovery_claim" not in checkpoint_after_throw
        assert app._session_control.has_active_tasks(session_id) is False
        assert invocation_scope._current_session_run_epoch(session_id) is None
        assert store.release_calls - releases_before == 1
        assert tool.calls == [{}]

    asyncio.run(scenario())


def test_cayu_app_recover_tool_round_athrow_preserves_cleanup_failure() -> None:
    class FailingRecoveryReleaseStore(FailingTerminalToolEventStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.fail_release = False

        async def release_session_invocation(self, command):
            if self.fail_release:
                self.fail_release = False
                try:
                    raise OSError("recovery database connection lost")
                except OSError as database_failure:
                    raise RuntimeError(
                        "manual recovery run fence release unavailable"
                    ) from database_failure
            return await super().release_session_invocation(command)

    session_id = "sess_tool_round_manual_athrow_cleanup_failure"
    store = FailingRecoveryReleaseStore()
    app, store, tool, checkpoint = _crashed_tool_round_app(
        session_id, store=store, tool=PureRecordingTool()
    )
    request = ToolRoundRecoveryRequest(
        session_id=session_id,
        round_id=checkpoint["pending_tool_round"]["tool_round_id"],
        tool_call_id="call_1",
        outcome=ToolApprovalRecoveryOutcome.COMPLETED,
        message="side effect completed externally",
    )

    async def scenario() -> None:
        stream = app.recover_tool_round(request)
        first_event = await asyncio.wait_for(anext(stream), timeout=5)
        assert first_event.type == EventType.SESSION_RESUMED

        store.fail_release = True
        injected = RuntimeError("consumer rejected recovery event")
        original_cause = ValueError("upstream consumer failure")
        injected.__cause__ = original_cause
        with pytest.raises(RuntimeError, match="consumer rejected recovery event") as exc_info:
            await stream.athrow(injected)
        assert exc_info.value is injected
        assert isinstance(injected.__cause__, BaseExceptionGroup)
        cleanup_failure, preserved_consumer_cause = injected.__cause__.exceptions
        assert isinstance(cleanup_failure, RuntimeError)
        assert str(cleanup_failure) == "manual recovery run fence release unavailable"
        assert isinstance(cleanup_failure.__cause__, OSError)
        assert str(cleanup_failure.__cause__) == "recovery database connection lost"
        assert preserved_consumer_cause is original_cause
        assert any(
            "Delegated runtime stream cleanup failed: RuntimeError" in note
            for note in getattr(injected, "__notes__", ())
        )

        interrupted = await store.load(session_id)
        assert interrupted is not None and interrupted.status == SessionStatus.INTERRUPTED
        checkpoint_after_throw = await store.load_checkpoint(session_id)
        assert checkpoint_after_throw is not None
        assert "incomplete_session_recovery_claim" not in checkpoint_after_throw
        assert app._session_control.has_active_tasks(session_id) is False
        assert invocation_scope._current_session_run_epoch(session_id) is None
        assert tool.calls == [{}]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "expire_acknowledgement",
    [False, True],
    ids=["live-claim", "expired-claim"],
)
def test_cayu_app_recover_tool_round_cancellation_after_claim_commit_preserves_cancel(
    monkeypatch: pytest.MonkeyPatch,
    expire_acknowledgement: bool,
):
    class BlockingRecoveryTransitionStore(FailingTerminalToolEventStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.recovery_transition_committed = asyncio.Event()
            self.allow_recovery_transition_return = asyncio.Event()

        async def transition_status_and_checkpoint(
            self,
            session_id: str,
            *,
            from_statuses: set[SessionStatus],
            to_status: SessionStatus,
            checkpoint_transform=None,
            store_time_checkpoint_transform=None,
            result_checkpoint_transform=None,
        ) -> Session:
            session = await super().transition_status_and_checkpoint(
                session_id,
                from_statuses=from_statuses,
                to_status=to_status,
                checkpoint_transform=checkpoint_transform,
                store_time_checkpoint_transform=store_time_checkpoint_transform,
                result_checkpoint_transform=result_checkpoint_transform,
            )
            if to_status == SessionStatus.RUNNING and SessionStatus.FAILED in from_statuses:
                self.recovery_transition_committed.set()
                await self.allow_recovery_transition_return.wait()
            return session

    async def scenario() -> None:
        session_id = "sess_tool_round_manual_claim_cancelled"
        store = BlockingRecoveryTransitionStore()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_1",
                        name="side_effect",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("unexpected continuation"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        tool = SideEffectTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

        initial_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
            ),
        )
        assert initial_events[-1].type == EventType.SESSION_FAILED
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        round_id = checkpoint["pending_tool_round"]["tool_round_id"]
        releases_before_recovery = store.release_calls
        if expire_acknowledgement:
            # Expire the local lease only after the claim commits. A short real
            # lease can run out before the commit on a loaded runner.
            from cayu.runtime import _recovery_claims

            real_monotonic = recovery_coordinator_module.time.monotonic
            monotonic_offset = 0.0

            class RecoveryMonotonicClock:
                @staticmethod
                def monotonic() -> float:
                    return real_monotonic() + monotonic_offset

            monkeypatch.setattr(recovery_coordinator_module, "time", RecoveryMonotonicClock)
            monkeypatch.setattr(_recovery_claims, "time", RecoveryMonotonicClock)
            cleanup_failure = RuntimeError("expired recovery claim cleanup failed")
            original_release = app._recovery_ownership.release_claim

            async def release_then_fail(session_id: str, claim_id: str) -> None:
                await original_release(session_id, claim_id)
                raise cleanup_failure

            monkeypatch.setattr(
                app._recovery_ownership,
                "release_claim",
                release_then_fail,
            )
        else:
            cleanup_failure = None

        recovery_task = asyncio.create_task(
            collect_tool_round_recovery_events(
                app,
                ToolRoundRecoveryRequest(
                    session_id=session_id,
                    round_id=round_id,
                    tool_call_id="call_1",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="side effect verified externally",
                ),
            )
        )
        await asyncio.wait_for(store.recovery_transition_committed.wait(), timeout=5)
        recovery_task.cancel("cancel committed manual recovery claim")
        assert recovery_task.cancelling() == 1
        if expire_acknowledgement:
            monotonic_offset = (
                recovery_ownership_module._INCOMPLETE_RECOVERY_CLAIM_LEASE.total_seconds() + 1
            )
        store.allow_recovery_transition_return.set()
        with pytest.raises(asyncio.CancelledError) as cancellation:
            await recovery_task
        assert cancellation.value.args == ("cancel committed manual recovery claim",)
        assert recovery_task.cancelled() is True
        if expire_acknowledgement:
            failure_graph = _runtime_exception_graph(cancellation.value)
            assert cleanup_failure is not None
            assert sum(candidate is cleanup_failure for candidate in failure_graph) == 1
            assert (
                sum(
                    isinstance(candidate, RuntimeError) and "consumed its lease" in str(candidate)
                    for candidate in failure_graph
                )
                == 1
            )

        session = await store.load(session_id)
        assert session is not None
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert checkpoint["pending_tool_round"]["tool_round_id"] == round_id
        assert "incomplete_session_recovery_claim" not in checkpoint
        events = await store.load_events(session_id)
        if expire_acknowledgement:
            assert session.status == SessionStatus.RUNNING
            assert not any(event.payload.get("manual_recovery") is True for event in events)
            assert store.release_calls == releases_before_recovery
        else:
            assert session.status == SessionStatus.INTERRUPTED
            assert events[-1].type == EventType.SESSION_INTERRUPTED
            assert events[-1].payload["abandoned"] is True
            assert store.release_calls - releases_before_recovery == 1
        assert tool.calls == [{}]
        assert app._session_control.has_active_tasks(session_id) is False

    asyncio.run(scenario())


def test_cayu_app_recover_tool_round_reconstruction_failure_releases_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "sess_tool_round_manual_reconstruction_failure"
    app, store, tool, checkpoint = _crashed_tool_round_app(session_id, tool=PureRecordingTool())
    request = ToolRoundRecoveryRequest(
        session_id=session_id,
        round_id=checkpoint["pending_tool_round"]["tool_round_id"],
        tool_call_id="call_1",
        outcome=ToolApprovalRecoveryOutcome.COMPLETED,
        message="side effect verified externally",
    )
    reconstruct_invocation_context = recovery_coordinator_module.reconstruct_invocation_context

    def fail_reconstruction(**_kwargs: Any) -> None:
        raise RuntimeError("manual recovery invocation reconstruction failed")

    monkeypatch.setattr(
        recovery_coordinator_module,
        "reconstruct_invocation_context",
        fail_reconstruction,
    )
    with pytest.raises(
        RuntimeError,
        match="manual recovery invocation reconstruction failed",
    ):
        asyncio.run(collect_tool_round_recovery_events(app, request))

    checkpoint_after_failure = asyncio.run(store.load_checkpoint(session_id))
    assert checkpoint_after_failure is not None
    assert "incomplete_session_recovery_claim" not in checkpoint_after_failure
    assert checkpoint_after_failure["pending_tool_round"]["tool_round_id"] == request.round_id
    assert invocation_scope._current_session_run_epoch(session_id) is None
    assert app._session_control.has_active_tasks(session_id) is False

    # Claim cleanup transfers unfinished durable work immediately. Restoring
    # the runtime context builder permits a same-process retry without waiting
    # for the former claim lease to expire.
    monkeypatch.setattr(
        recovery_coordinator_module,
        "reconstruct_invocation_context",
        reconstruct_invocation_context,
    )
    recovered = asyncio.run(collect_tool_round_recovery_events(app, request))
    assert recovered[-1].type == EventType.SESSION_COMPLETED
    assert tool.calls == [{}]


def test_cayu_app_recover_tool_round_reconciles_ambiguous_claim_commit() -> None:
    class CommitThenRaiseRecoveryClaimStore(FailingTerminalToolEventStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.failed_manual_claim_ack = False

        async def transition_status_and_checkpoint(
            self,
            session_id: str,
            *,
            from_statuses: set[SessionStatus],
            to_status: SessionStatus,
            checkpoint_transform=None,
            store_time_checkpoint_transform=None,
            result_checkpoint_transform=None,
        ) -> Session:
            transitioned = await super().transition_status_and_checkpoint(
                session_id,
                from_statuses=from_statuses,
                to_status=to_status,
                checkpoint_transform=checkpoint_transform,
                store_time_checkpoint_transform=store_time_checkpoint_transform,
                result_checkpoint_transform=result_checkpoint_transform,
            )
            checkpoint = await self.load_checkpoint(session_id)
            marker = (
                None if checkpoint is None else checkpoint.get("incomplete_session_recovery_claim")
            )
            if (
                not self.failed_manual_claim_ack
                and type(marker) is dict
                and marker.get("operation") == "manual_tool_round_recovery"
            ):
                self.failed_manual_claim_ack = True
                raise RuntimeError("manual recovery claim commit acknowledgement lost")
            return transitioned

    session_id = "sess_tool_round_manual_claim_ack_lost"
    store = CommitThenRaiseRecoveryClaimStore()
    app, store, tool, checkpoint = _crashed_tool_round_app(
        session_id, store=store, tool=PureRecordingTool()
    )
    request = ToolRoundRecoveryRequest(
        session_id=session_id,
        round_id=checkpoint["pending_tool_round"]["tool_round_id"],
        tool_call_id="call_1",
        outcome=ToolApprovalRecoveryOutcome.COMPLETED,
        message="verified externally",
    )

    recovered = asyncio.run(collect_tool_round_recovery_events(app, request))
    assert store.failed_manual_claim_ack is True
    assert recovered[-1].type == EventType.SESSION_COMPLETED
    checkpoint_after_recovery = asyncio.run(store.load_checkpoint(session_id))
    assert checkpoint_after_recovery is not None
    assert "incomplete_session_recovery_claim" not in checkpoint_after_recovery
    assert tool.calls == [{}]


def test_cayu_app_recover_tool_round_preserves_cancellation_during_claim_reconciliation() -> None:
    class BlockingClaimReconciliationStore(FailingTerminalToolEventStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.failed_manual_claim_ack = False
            self.reconciliation_started = asyncio.Event()
            self.allow_reconciliation = asyncio.Event()

        async def transition_status_and_checkpoint(
            self,
            session_id: str,
            *,
            from_statuses: set[SessionStatus],
            to_status: SessionStatus,
            checkpoint_transform=None,
            store_time_checkpoint_transform=None,
            result_checkpoint_transform=None,
        ) -> Session:
            transitioned = await super().transition_status_and_checkpoint(
                session_id,
                from_statuses=from_statuses,
                to_status=to_status,
                checkpoint_transform=checkpoint_transform,
                store_time_checkpoint_transform=store_time_checkpoint_transform,
                result_checkpoint_transform=result_checkpoint_transform,
            )
            checkpoint = await super().load_checkpoint(session_id)
            marker = (
                None if checkpoint is None else checkpoint.get("incomplete_session_recovery_claim")
            )
            if (
                not self.failed_manual_claim_ack
                and type(marker) is dict
                and marker.get("operation") == "manual_tool_round_recovery"
            ):
                self.failed_manual_claim_ack = True
                raise RuntimeError("manual recovery claim commit acknowledgement lost")
            return transitioned

        async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
            if self.failed_manual_claim_ack and not self.reconciliation_started.is_set():
                self.reconciliation_started.set()
                await self.allow_reconciliation.wait()
            return await super().load_checkpoint(session_id)

    session_id = "sess_tool_round_manual_claim_reconciliation_cancelled"
    store = BlockingClaimReconciliationStore()
    app, store, tool, checkpoint = _crashed_tool_round_app(
        session_id, store=store, tool=PureRecordingTool()
    )
    request = ToolRoundRecoveryRequest(
        session_id=session_id,
        round_id=checkpoint["pending_tool_round"]["tool_round_id"],
        tool_call_id="call_1",
        outcome=ToolApprovalRecoveryOutcome.COMPLETED,
        message="verified externally",
    )

    async def scenario() -> None:
        recovery_task = asyncio.create_task(collect_tool_round_recovery_events(app, request))
        await asyncio.wait_for(store.reconciliation_started.wait(), timeout=10)
        assert recovery_task.cancelling() == 0
        recovery_task.cancel("cancel during manual claim reconciliation")
        assert recovery_task.cancelling() == 1
        await asyncio.sleep(0)
        assert recovery_task.done() is False
        store.allow_reconciliation.set()
        with pytest.raises(asyncio.CancelledError) as cancellation:
            await asyncio.wait_for(recovery_task, timeout=5)
        assert cancellation.value.args == ("cancel during manual claim reconciliation",)
        assert recovery_task.cancelled()

        interrupted = await store.load(session_id)
        assert interrupted is not None and interrupted.status == SessionStatus.INTERRUPTED
        checkpoint_after_failure = await store.load_checkpoint(session_id)
        assert checkpoint_after_failure is not None
        assert "pending_tool_round" in checkpoint_after_failure
        assert "incomplete_session_recovery_claim" not in checkpoint_after_failure
        assert not any(
            event.payload.get("manual_recovery") is True
            for event in await store.load_events(session_id)
        )

        recovered = await collect_tool_round_recovery_events(app, request)
        assert recovered[-1].type == EventType.SESSION_COMPLETED
        assert tool.calls == [{}]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("delayed_boundary", "expected_expired_fence_failures"),
    [
        ("initial_claim", 0),
        ("interruption_fence", 1),
    ],
)
def test_manual_recovery_interruption_fence_uses_its_own_lease_deadline(
    monkeypatch: pytest.MonkeyPatch,
    delayed_boundary: str,
    expected_expired_fence_failures: int,
) -> None:
    class DelayedInterruptionFenceStore(FailingTerminalToolEventStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.pause_manual_claim = False
            self.manual_claim_started = asyncio.Event()
            self.allow_manual_claim = asyncio.Event()
            self.operator_interrupt_committed = asyncio.Event()
            self.interruption_fence_committed = asyncio.Event()
            self.allow_interruption_fence_return = asyncio.Event()

        async def transition_status_and_checkpoint(
            self,
            session_id: str,
            *,
            from_statuses: set[SessionStatus],
            to_status: SessionStatus,
            checkpoint_transform=None,
            store_time_checkpoint_transform=None,
            result_checkpoint_transform=None,
        ) -> Session:
            if self.pause_manual_claim and to_status == SessionStatus.RUNNING:
                self.manual_claim_started.set()
                await self.allow_manual_claim.wait()
            session = await super().transition_status_and_checkpoint(
                session_id,
                from_statuses=from_statuses,
                to_status=to_status,
                checkpoint_transform=checkpoint_transform,
                store_time_checkpoint_transform=store_time_checkpoint_transform,
                result_checkpoint_transform=result_checkpoint_transform,
            )
            if to_status == SessionStatus.INTERRUPTING:
                self.operator_interrupt_committed.set()
            return session

        async def fence_run_and_transform_checkpoint(self, *args, **kwargs) -> Session:
            fenced = await super().fence_run_and_transform_checkpoint(*args, **kwargs)
            checkpoint = await super().load_checkpoint(fenced.id)
            marker = (
                None if checkpoint is None else checkpoint.get("incomplete_session_recovery_claim")
            )
            if (
                type(marker) is dict
                and marker.get("operation") == "manual_tool_round_interruption_fence"
            ):
                self.interruption_fence_committed.set()
                await self.allow_interruption_fence_return.wait()
            return fenced

    session_id = f"sess_tool_round_interruption_fence_{delayed_boundary}_cancelled"
    store = DelayedInterruptionFenceStore()
    app, store, tool, checkpoint = _crashed_tool_round_app(session_id, store=store)
    operator_app = CayuApp(session_store=store, enable_logging=False)
    operator_app.register_provider(FakeProvider([]), default=True)
    operator_app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    active_profile = execution_profiles_module.active_invocation_execution_profile_from_checkpoint(
        checkpoint
    )
    assert active_profile is not None
    request = ToolRoundRecoveryRequest(
        session_id=session_id,
        round_id=checkpoint["pending_tool_round"]["tool_round_id"],
        tool_call_id="call_1",
        outcome=ToolApprovalRecoveryOutcome.COMPLETED,
        message="side effect verified externally",
    )
    real_monotonic = recovery_coordinator_module.time.monotonic
    monotonic_offset = 0.0

    class RecoveryMonotonicClock:
        @staticmethod
        def monotonic() -> float:
            return real_monotonic() + monotonic_offset

    monkeypatch.setattr(
        recovery_coordinator_module,
        "time",
        RecoveryMonotonicClock,
    )
    from cayu.runtime import _recovery_claims

    monkeypatch.setattr(_recovery_claims, "time", RecoveryMonotonicClock)
    monkeypatch.setattr(
        recovery_ownership_module,
        "_INCOMPLETE_RECOVERY_CLAIM_LEASE",
        timedelta(seconds=30),
    )

    async def scenario() -> None:
        nonlocal monotonic_offset
        current = await store.load(session_id)
        assert current is not None
        await rebind_test_invocation(
            store,
            session_id,
            target_status=SessionStatus.RUNNING,
            checkpoint_transform=lambda session, current_checkpoint: (
                execution_profiles_module.checkpoint_with_active_invocation_execution_profile(
                    current_checkpoint,
                    session_id=session.id,
                    interaction_id=active_profile.interaction_id,
                    run_epoch=session.run_epoch + 1,
                    profile=active_profile.profile,
                    expected=active_profile,
                )
            ),
        )
        await store.release_run_fence(session_id)
        store.pause_manual_claim = True
        recovery_task = asyncio.create_task(collect_tool_round_recovery_events(app, request))
        await asyncio.wait_for(store.manual_claim_started.wait(), timeout=10)

        interruption_task = asyncio.create_task(
            collect_interrupt_events(
                operator_app,
                InterruptSessionRequest(
                    session_id=session_id,
                    reason="operator superseded manual recovery",
                ),
            )
        )
        await asyncio.wait_for(store.operator_interrupt_committed.wait(), timeout=5)
        recovery_task.cancel("cancel superseded manual recovery")
        assert recovery_task.cancelling() == 1
        if delayed_boundary == "initial_claim":
            monotonic_offset += 31
        store.allow_manual_claim.set()
        await asyncio.wait_for(store.interruption_fence_committed.wait(), timeout=5)
        if delayed_boundary == "interruption_fence":
            monotonic_offset += 31
        store.allow_interruption_fence_return.set()

        with pytest.raises(asyncio.CancelledError) as cancellation:
            await recovery_task
        assert cancellation.value.args == ("cancel superseded manual recovery",)
        assert recovery_task.cancelled() is True
        failure_graph = _runtime_exception_graph(cancellation.value)
        assert (
            sum(
                isinstance(candidate, RuntimeError) and "consumed its lease" in str(candidate)
                for candidate in failure_graph
            )
            == expected_expired_fence_failures
        ), [f"{type(candidate).__name__}: {candidate}" for candidate in failure_graph]

        await operator_app._session_finalization.finalize_abandoned_session_by_id(session_id)
        interruption_events = await asyncio.wait_for(interruption_task, timeout=5)
        assert interruption_events[-1].type == EventType.SESSION_INTERRUPTED
        interrupted = await store.load(session_id)
        assert interrupted is not None and interrupted.status == SessionStatus.INTERRUPTED
        checkpoint_after_cancel = await store.load_checkpoint(session_id)
        assert checkpoint_after_cancel is not None
        assert "incomplete_session_recovery_claim" not in checkpoint_after_cancel
        assert tool.calls == [{}]
        assert await operator_app.drain_background_interruptions(timeout_s=1) is True
        assert await app.drain_background_interruptions(timeout_s=1) is True

    asyncio.run(scenario())
