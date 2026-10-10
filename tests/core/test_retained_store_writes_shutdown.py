"""Shutdown waits for store writes that outlive their owner's bounded wait."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from cayu.applications import CayuApp
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.runtime import _recovery_ownership as recovery_ownership_module
from cayu.sessions.base import InMemorySessionStore
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.requests import RunRequest


class _ClosableSessionStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.closed = False
        self.writes_after_close: list[str] = []

    async def close(self) -> None:
        self.closed = True

    async def transform_checkpoint_with_store_time(self, session_id, transform, *args, **kwargs):
        if self.closed:
            self.writes_after_close.append(session_id)
        return await super().transform_checkpoint_with_store_time(
            session_id, transform, *args, **kwargs
        )


async def _assert_no_claim_heartbeat_runs(app: CayuApp) -> None:
    owner = app._model_step_executor._provider_operation_cancellation
    async with asyncio.timeout(2):
        while any(not task.done() for task in owner.running()):
            await asyncio.sleep(0.01)


def test_shutdown_waits_for_a_detached_interruption_claim_renewal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.runtime import _interruption_coordinator as coordinator_module

    monkeypatch.setattr(coordinator_module, "_BACKGROUND_INTERRUPTION_LEASE_SECONDS", 0.5)
    monkeypatch.setattr(coordinator_module, "_BACKGROUND_INTERRUPTION_HEARTBEAT_SECONDS", 0.01)

    async def scenario() -> None:
        store = _ClosableSessionStore()
        app = CayuApp(session_store=store, enable_logging=False, owned_resources=(store,))
        identity = SessionIdentity(provider_name="fake", model="fake-model")
        for session_id, parent, metadata in (
            ("parent", None, {}),
            ("child", "parent", {"subagent": {"mode": "background"}}),
        ):
            await store.create(
                RunRequest(
                    agent_name="r",
                    session_id=session_id,
                    parent_session_id=parent,
                    messages=[Message.text("user", "t")],
                    metadata=metadata,
                ),
                identity=identity,
            )
        coordinator = app._background_interruption_coordinator
        renewal_started, release = asyncio.Event(), asyncio.Event()
        renew = coordinator._renew_pending_interruption_cascade_claim

        async def blocked_renewal(*args):
            # A renewal write that outlives the claim's deadline.
            renewal_started.set()
            await release.wait()
            return await renew(*args)

        async def interrupt_child(request):
            await renewal_started.wait()
            await store.update_status(request.session_id, SessionStatus.INTERRUPTED)
            yield Event(
                type=EventType.SESSION_INTERRUPTED,
                session_id=request.session_id,
                agent_name="r",
            )

        monkeypatch.setattr(
            coordinator, "_renew_pending_interruption_cascade_claim", blocked_renewal
        )
        monkeypatch.setattr(coordinator, "_interrupt_session", interrupt_child)
        cascade = app._session_finalization.schedule_background_interruption_cascade(
            parent_session_id="parent",
            interrupt_payload={
                "reason": "stop",
                "metadata": {},
                "requested_by": None,
                "interruption_type": "operator_requested",
            },
            create_if_missing=True,
        )
        assert cascade is not None
        await asyncio.wait_for(asyncio.shield(cascade), 5)
        try:
            outcome = await app.aclose(timeout_s=1)
            assert not outcome.settled
            assert outcome.owned_resources == "retained" and not store.closed
        finally:
            release.set()
        assert (await app.aclose(timeout_s=5)).settled and store.closed
        assert store.writes_after_close == []

    asyncio.run(scenario())


@pytest.mark.parametrize("released", ["after_shutdown", "within_budget"])
def test_shutdown_waits_for_a_detached_provider_cancellation_renewal(
    monkeypatch: pytest.MonkeyPatch, released: str
) -> None:
    from datetime import timedelta

    from tests.core.test_provider_operation_offline_recovery import (
        _BlockingCancellationProvider,
        _stage_offline_operation,
    )

    import cayu.runtime._provider_operation_cancellation_owner as cancellation_owner
    from cayu.agents import AgentSpec
    from cayu.sessions.requests import InterruptSessionRequest

    monkeypatch.setattr(
        cancellation_owner,
        "_PROVIDER_OPERATION_CANCELLATION_CLAIM_LEASE",
        timedelta(milliseconds=500),
    )
    monkeypatch.setattr(
        cancellation_owner, "_PROVIDER_OPERATION_CANCELLATION_CLAIM_HEARTBEAT_SECONDS", 0.005
    )

    class HangingRenewalStore(_ClosableSessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.armed = False
            self.hung = asyncio.Event()
            self.release = asyncio.Event()

        async def publish_checkpoint_and_events_with_store_time(self, session_id, **kwargs):
            events = kwargs["events"]
            if any(event.type is EventType.PROVIDER_OPERATION_CANCEL_REQUESTED for event in events):
                self.armed = True
            elif self.armed and not events and not self.hung.is_set():
                # A claim renewal write that outlives its lease.
                self.hung.set()
                await self.release.wait()
                if self.closed:
                    self.writes_after_close.append(kwargs["idempotency_key"])
            return await super().publish_checkpoint_and_events_with_store_time(session_id, **kwargs)

    async def scenario() -> None:
        store = HangingRenewalStore()
        provider = _BlockingCancellationProvider()
        await _stage_offline_operation(store, session_id="cancel-renewal", provider=provider)
        app = CayuApp(session_store=store, enable_logging=False, owned_resources=(store,))
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        async def interrupt() -> None:
            async for _ in app.interrupt_session(
                InterruptSessionRequest(session_id="cancel-renewal", reason="stop")
            ):
                pass

        interrupting = asyncio.create_task(interrupt())
        await asyncio.wait_for(store.hung.wait(), 5)
        await asyncio.gather(asyncio.wait_for(interrupting, 5), return_exceptions=True)
        if released == "within_budget":
            # The write lands while shutdown waits: the first attempt settles.
            asyncio.get_running_loop().call_later(0.2, store.release.set)
            outcome = await app.aclose(timeout_s=5)
            assert outcome.settled and store.closed
            assert store.writes_after_close == []
            return
        try:
            # The public drain also reports the renewal still writing.
            assert await app.drain_provider_operation_cancellations(timeout_s=0.3) is False
            outcome = await app.aclose(timeout_s=1)
            assert not outcome.settled
            assert outcome.owned_resources == "retained" and not store.closed
        finally:
            store.release.set()
        assert (await app.aclose(timeout_s=5)).settled and store.closed
        assert store.writes_after_close == []

    asyncio.run(scenario())


def test_shutdown_waits_for_a_detached_recovery_claim_renewal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time

    from cayu.runtime import _recovery_coordinator as recovery_module

    monkeypatch.setattr(
        recovery_ownership_module, "_INCOMPLETE_RECOVERY_CLAIM_HEARTBEAT_INTERVAL_SECONDS", 0.01
    )

    class BlockingRenewalStore(_ClosableSessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.block_renewal = False
            self.renewal_dispatched = asyncio.Event()
            self.release = asyncio.Event()

        async def transform_checkpoint_with_store_time(
            self, session_id, transform, *args, **kwargs
        ):
            if self.block_renewal:
                self.renewal_dispatched.set()
                await self.release.wait()
            return await super().transform_checkpoint_with_store_time(
                session_id, transform, *args, **kwargs
            )

    async def scenario() -> None:
        store = BlockingRenewalStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="recovery-renewal",
                messages=[Message.text("user", "recover")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        app = CayuApp(session_store=store, enable_logging=False, owned_resources=(store,))
        coordinator = app._recovery_coordinator
        claim = await coordinator._recovery_ownership.claim(
            session=session, inactive_for_seconds=None
        )
        assert claim is not None
        store.block_renewal = True
        # The renewal outlives the claim's local deadline and is detached.
        with pytest.raises(recovery_module._IncompleteRecoveryClaimLost):
            await coordinator._recovery_ownership.heartbeat(
                session_id=session.id,
                claim_id=claim.claim_id,
                local_lease_deadline=time.monotonic() + 0.3,
                stop=asyncio.Event(),
            )
        assert store.renewal_dispatched.is_set()
        try:
            outcome = await app.aclose(timeout_s=1)
            assert not outcome.settled
            assert outcome.owned_resources == "retained" and not store.closed
        finally:
            store.release.set()
        assert (await app.aclose(timeout_s=5)).settled and store.closed
        assert store.writes_after_close == []

    asyncio.run(scenario())


def test_shutdown_waits_for_a_timed_out_tool_result_projection(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from tests.core.test_tool_result_projection import _collect, _FakeProvider, _ResultTool

    from cayu.agents import AgentSpec
    from cayu.artifacts.local import LocalArtifactStore
    from cayu.environments.base import Environment, EnvironmentSpec
    from cayu.providers.base import ModelStreamEvent
    from cayu.runtime._tool_invocation import terminal as invocation_terminal
    from cayu.tools.base import ToolResult
    from cayu.tools.result_projection import ArtifactExternalizingToolResultPolicy

    monkeypatch.setattr(invocation_terminal, "_TOOL_RESULT_PROJECTION_TIMEOUT_SECONDS", 0.05)

    class OwnedSlowArtifactStore(LocalArtifactStore):
        """Its write ignores cancellation, outliving the projection timeout."""

        def __init__(self, root, *, store_id) -> None:
            super().__init__(root, store_id=store_id)
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.closed = False
            self.written_after_close: bool | None = None

        async def put_bytes(self, content, *, filename, **kwargs):
            self.started.set()
            while not self.release.is_set():
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    task = asyncio.current_task()
                    assert task is not None
                    while task.cancelling():
                        task.uncancel()
            self.written_after_close = self.closed
            return await super().put_bytes(content, filename=filename, **kwargs)

        async def close(self) -> None:
            self.closed = True

    async def scenario() -> None:
        store = OwnedSlowArtifactStore(tmp_path / "artifacts", store_id="owned")
        provider = _FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(id="call", name="result_tool", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
            ]
        )
        app = CayuApp(
            enable_logging=False,
            owned_resources=(store,),
            tool_result_projection_policy=ArtifactExternalizingToolResultPolicy(
                max_inline_bytes=256, max_inline_token_estimate=None
            ),
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), artifact_store=store), default=True
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[_ResultTool(ToolResult(content="x" * 10_000))],
        )
        await asyncio.wait_for(
            _collect(
                app.run(
                    RunRequest(
                        session_id="projection",
                        agent_name="assistant",
                        messages=[Message.text("user", "run")],
                    )
                )
            ),
            30,
        )
        assert store.started.is_set()
        try:
            outcome = await app.aclose(timeout_s=1)
            step = outcome.step("environment_cleanups")
            assert step is not None and step.status == "incomplete"
            assert outcome.owned_resources == "retained" and not store.closed
        finally:
            store.release.set()
        assert (await app.aclose(timeout_s=5)).settled and store.closed
        assert store.written_after_close is False

    asyncio.run(scenario())


def test_a_failed_provider_cancellation_stops_its_heartbeat_for_shutdown() -> None:
    from tests.core.test_provider_operation_offline_recovery import (
        _CompletionWinsCancellationProvider,
        _stage_offline_operation,
    )

    from cayu.agents import AgentSpec
    from cayu.sessions.requests import InterruptSessionRequest

    class FailingResolutionStore(_ClosableSessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        async def publish_checkpoint_and_events_with_store_time(self, session_id, **kwargs):
            if not self.failed and any(
                event.type is EventType.PROVIDER_OPERATION_CANCEL_RESOLVED
                for event in kwargs["events"]
            ):
                # The cancellation fails after its claim heartbeat started.
                self.failed = True
                raise RuntimeError("transient store failure")
            return await super().publish_checkpoint_and_events_with_store_time(session_id, **kwargs)

    async def scenario() -> None:
        store = FailingResolutionStore()
        provider = _CompletionWinsCancellationProvider()
        await _stage_offline_operation(store, session_id="failed-cancel", provider=provider)
        app = CayuApp(session_store=store, enable_logging=False, owned_resources=(store,))
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        # The interrupt runs in this task, which then shuts the app down.
        with contextlib.suppress(Exception):
            async for _ in app.interrupt_session(
                InterruptSessionRequest(session_id="failed-cancel", reason="stop")
            ):
                pass
        # The failed cancellation stopped its heartbeat already, not only at shutdown.
        await _assert_no_claim_heartbeat_runs(app)
        outcome = await app.aclose(timeout_s=5)
        assert outcome.settled and store.closed
        assert store.writes_after_close == []

    asyncio.run(scenario())


def test_a_cascade_cancelled_by_a_drain_timeout_is_waited_for_on_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.runtime import _interruption_coordinator as coordinator_module

    monkeypatch.setattr(coordinator_module, "_BACKGROUND_INTERRUPTION_LEASE_SECONDS", 0.5)
    monkeypatch.setattr(coordinator_module, "_BACKGROUND_INTERRUPTION_HEARTBEAT_SECONDS", 0.01)

    async def scenario() -> None:
        store = _ClosableSessionStore()
        app = CayuApp(session_store=store, enable_logging=False, owned_resources=(store,))
        identity = SessionIdentity(provider_name="fake", model="fake-model")
        for session_id, parent, metadata in (
            ("parent", None, {}),
            ("child", "parent", {"subagent": {"mode": "background"}}),
        ):
            await store.create(
                RunRequest(
                    agent_name="r",
                    session_id=session_id,
                    parent_session_id=parent,
                    messages=[Message.text("user", "t")],
                    metadata=metadata,
                ),
                identity=identity,
            )
        coordinator = app._background_interruption_coordinator
        renewal_started, release = asyncio.Event(), asyncio.Event()
        renew = coordinator._renew_pending_interruption_cascade_claim
        release_claim = coordinator._release_pending_interruption_cascade_claim
        writes: list[bool] = []

        async def blocked_renewal(*args):
            renewal_started.set()
            await release.wait()
            await asyncio.sleep(0.05)  # a slow store write
            writes.append(store.closed)
            return await renew(*args)

        async def recorded_release(*args):
            writes.append(store.closed)
            return await release_claim(*args)

        async def interrupt_child(request):
            await renewal_started.wait()
            await asyncio.sleep(10)  # still interrupting when the drain times out
            yield Event(
                type=EventType.SESSION_INTERRUPTED,
                session_id=request.session_id,
                agent_name="r",
            )

        monkeypatch.setattr(
            coordinator, "_renew_pending_interruption_cascade_claim", blocked_renewal
        )
        monkeypatch.setattr(
            coordinator, "_release_pending_interruption_cascade_claim", recorded_release
        )
        monkeypatch.setattr(coordinator, "_interrupt_session", interrupt_child)
        app._session_finalization.schedule_background_interruption_cascade(
            parent_session_id="parent",
            interrupt_payload={
                "reason": "stop",
                "metadata": {},
                "requested_by": None,
                "interruption_type": "operator_requested",
            },
            create_if_missing=True,
        )
        await asyncio.wait_for(renewal_started.wait(), 5)
        try:
            first = await app.aclose(timeout_s=0.3)
            assert not first.settled and not store.closed
        finally:
            release.set()
        # The cancelled cascade still renews and releases its claim; the retry
        # waits for both before closing the store.
        assert (await app.aclose(timeout_s=5)).settled and store.closed
        assert writes and not any(writes)

    asyncio.run(scenario())


def test_a_hung_projection_does_not_skip_environment_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        app = CayuApp(enable_logging=False)
        release = asyncio.Event()
        app._tool_round_executor.invocation.terminals._retain_detached_projection(
            asyncio.create_task(release.wait())
        )
        cleaned: list[float] = []
        drain_retained = app._environment_lifecycle.drain_retained_cleanups

        async def recorded_drain(*, timeout_s: float) -> bool:
            cleaned.append(timeout_s)
            return await drain_retained(timeout_s=timeout_s)

        monkeypatch.setattr(app._environment_lifecycle, "drain_retained_cleanups", recorded_drain)
        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            assert await app.drain_environment_cleanups(timeout_s=0.3) is False
            # One shared deadline, and environment cleanup still ran.
            assert loop.time() - started < 1.0
            assert cleaned
        finally:
            release.set()
        assert await app.drain_environment_cleanups(timeout_s=5) is True

    asyncio.run(scenario())


def test_a_failed_claim_release_after_a_won_completion_does_not_stall_shutdown() -> None:
    from tests.core.test_provider_operation_offline_recovery import (
        _CompletionWinsCancellationProvider,
        _stage_offline_operation,
    )

    from cayu.agents import AgentSpec
    from cayu.sessions.requests import InterruptSessionRequest

    class FailingReleaseStore(_ClosableSessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.armed = False
            self.failed = False

        async def promote_model_completion_stage(self, *args, **kwargs):
            promoted = await super().promote_model_completion_stage(*args, **kwargs)
            self.armed = True
            return promoted

        async def publish_checkpoint_and_events_with_store_time(self, session_id, **kwargs):
            if self.armed and not kwargs["events"] and not self.failed:
                # The claim release after recovery fails once.
                self.failed = True
                raise RuntimeError("transient store failure on claim release")
            return await super().publish_checkpoint_and_events_with_store_time(session_id, **kwargs)

    async def scenario() -> None:
        store = FailingReleaseStore()
        provider = _CompletionWinsCancellationProvider()
        await _stage_offline_operation(store, session_id="won-completion", provider=provider)
        app = CayuApp(session_store=store, enable_logging=False, owned_resources=(store,))
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        with contextlib.suppress(Exception):
            async for _ in app.interrupt_session(
                InterruptSessionRequest(session_id="won-completion", reason="stop")
            ):
                pass
        assert store.failed
        # The failed release stopped its heartbeat already, not only at shutdown.
        await _assert_no_claim_heartbeat_runs(app)
        # This task held the claim; its shutdown must not wait on the heartbeat.
        outcome = await app.aclose(timeout_s=5)
        assert outcome.settled and store.closed
        assert store.writes_after_close == []

    asyncio.run(scenario())


def test_a_worker_still_writing_after_a_drain_timeout_is_waited_for_on_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.runtime import _interruption_coordinator as coordinator_module

    monkeypatch.setattr(coordinator_module, "_BACKGROUND_INTERRUPTION_LEASE_SECONDS", 0.5)
    monkeypatch.setattr(coordinator_module, "_BACKGROUND_INTERRUPTION_HEARTBEAT_SECONDS", 0.01)

    async def scenario() -> None:
        store = _ClosableSessionStore()
        app = CayuApp(session_store=store, enable_logging=False, owned_resources=(store,))
        identity = SessionIdentity(provider_name="fake", model="fake-model")
        for session_id, parent, metadata in (
            ("parent", None, {}),
            ("child", "parent", {"subagent": {"mode": "background"}}),
        ):
            await store.create(
                RunRequest(
                    agent_name="r",
                    session_id=session_id,
                    parent_session_id=parent,
                    messages=[Message.text("user", "t")],
                    metadata=metadata,
                ),
                identity=identity,
            )
        coordinator = app._background_interruption_coordinator
        interrupting, release = asyncio.Event(), asyncio.Event()
        writes: list[bool] = []

        async def interrupt_child(request):
            interrupting.set()
            try:
                await asyncio.sleep(10)  # still interrupting when the drain times out
            except asyncio.CancelledError:
                # A durable interruption write that finishes after cancellation.
                await release.wait()
                await asyncio.sleep(0.05)
                writes.append(store.closed)
                raise
            yield Event(
                type=EventType.SESSION_INTERRUPTED,
                session_id=request.session_id,
                agent_name="r",
            )

        monkeypatch.setattr(coordinator, "_interrupt_session", interrupt_child)
        app._session_finalization.schedule_background_interruption_cascade(
            parent_session_id="parent",
            interrupt_payload={
                "reason": "stop",
                "metadata": {},
                "requested_by": None,
                "interruption_type": "operator_requested",
            },
            create_if_missing=True,
        )
        await asyncio.wait_for(interrupting.wait(), 5)
        try:
            first = await app.aclose(timeout_s=0.3)
            assert not first.settled and not store.closed
        finally:
            release.set()
        assert (await app.aclose(timeout_s=5)).settled and store.closed
        assert writes == [False]

    asyncio.run(scenario())


def _register_orphaned_heartbeat(app: CayuApp):
    from cayu.runtime._provider_operation_cancellation_owner import (
        _ProviderOperationCancellationHeartbeat,
    )

    owner = app._model_step_executor._provider_operation_cancellation
    control = _ProviderOperationCancellationHeartbeat(
        stop=asyncio.Event(), release_intended=asyncio.Event(), claim_deadline_monotonic=0.0
    )

    async def heartbeat() -> None:
        # Renews until stopped, like a claim heartbeat whose holder moved on.
        await control.stop.wait()

    control.task = asyncio.create_task(heartbeat())
    owner._provider_operation_cancellation_heartbeats["orphaned-claim"] = control
    return control


@pytest.mark.parametrize("in_flight", [False, True])
def test_shutdown_stops_an_orphaned_claim_heartbeat_only_when_nothing_runs(
    in_flight: bool,
) -> None:
    async def scenario() -> None:
        store = _ClosableSessionStore()
        app = CayuApp(session_store=store, enable_logging=False, owned_resources=(store,))
        control = _register_orphaned_heartbeat(app)
        release = asyncio.Event()
        operation = None
        if in_flight:

            async def running() -> None:
                await release.wait()

            operation = asyncio.create_task(app._run_worker_step(running))
            await asyncio.sleep(0)
        try:
            outcome = await app.aclose(timeout_s=0.5)
            if not in_flight:
                assert outcome.settled and store.closed and control.stop.is_set()
                return
            # A running operation may still need its claim, so it is left alone.
            assert not outcome.settled and not store.closed
            assert not control.stop.is_set()
        finally:
            release.set()
            if operation is not None:
                await operation
        assert (await app.aclose(timeout_s=5)).settled and store.closed

    asyncio.run(scenario())


def test_a_recovery_renewal_settling_in_budget_lets_the_first_attempt_settle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time

    from cayu.runtime import _recovery_coordinator as recovery_module

    monkeypatch.setattr(
        recovery_ownership_module, "_INCOMPLETE_RECOVERY_CLAIM_HEARTBEAT_INTERVAL_SECONDS", 0.01
    )

    class BlockingRenewalStore(_ClosableSessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.block_renewal = False
            self.release = asyncio.Event()

        async def transform_checkpoint_with_store_time(
            self, session_id, transform, *args, **kwargs
        ):
            if self.block_renewal:
                await self.release.wait()
            return await super().transform_checkpoint_with_store_time(
                session_id, transform, *args, **kwargs
            )

    async def scenario() -> None:
        store = BlockingRenewalStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="recovery-in-budget",
                messages=[Message.text("user", "recover")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        app = CayuApp(session_store=store, enable_logging=False, owned_resources=(store,))
        coordinator = app._recovery_coordinator
        claim = await coordinator._recovery_ownership.claim(
            session=session, inactive_for_seconds=None
        )
        assert claim is not None
        store.block_renewal = True
        with pytest.raises(recovery_module._IncompleteRecoveryClaimLost):
            await coordinator._recovery_ownership.heartbeat(
                session_id=session.id,
                claim_id=claim.claim_id,
                local_lease_deadline=time.monotonic() + 0.3,
                stop=asyncio.Event(),
            )
        # The write lands while shutdown waits: the first attempt settles.
        asyncio.get_running_loop().call_later(0.2, store.release.set)
        outcome = await app.aclose(timeout_s=5)
        assert outcome.settled and store.closed
        assert store.writes_after_close == []

    asyncio.run(scenario())


@pytest.mark.parametrize("timeout_s", ["1", None, 0, -1.0, float("nan"), True])
def test_the_provider_cancellation_drain_validates_its_timeout(timeout_s: object) -> None:
    async def scenario() -> None:
        app = CayuApp(enable_logging=False)
        with pytest.raises(ValueError, match="finite positive"):
            await app.drain_provider_operation_cancellations(timeout_s=timeout_s)  # ty: ignore[invalid-argument-type]

    asyncio.run(scenario())


def test_a_failed_claim_release_keeps_its_error_over_a_failing_renewal() -> None:
    from types import SimpleNamespace

    class ReleaseFailed(RuntimeError):
        pass

    class FailingReleaseStore(_ClosableSessionStore):
        async def publish_checkpoint_and_events_with_store_time(self, session_id, **kwargs):
            raise ReleaseFailed("release write failed")

    async def scenario() -> None:
        app = CayuApp(session_store=FailingReleaseStore(), enable_logging=False)
        control = _register_orphaned_heartbeat(app)
        heartbeat = control.task
        assert heartbeat is not None
        heartbeat.cancel()

        async def failing_renewal() -> None:
            await control.stop.wait()
            raise RuntimeError("renewal failed")

        control.task = asyncio.create_task(failing_renewal())
        await asyncio.gather(heartbeat, return_exceptions=True)
        owner = app._model_step_executor._provider_operation_cancellation
        with pytest.raises(ReleaseFailed):
            await owner.release_claim(
                session=SimpleNamespace(id="session", run_epoch=1),  # ty: ignore[invalid-argument-type]
                claim=SimpleNamespace(claim_id="orphaned-claim"),  # ty: ignore[invalid-argument-type]
            )
        assert control.task.done()

    asyncio.run(scenario())


def test_cancelling_the_drain_during_its_cleanup_keeps_still_writing_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.runtime import _interruption_coordinator as coordinator_module

    monkeypatch.setattr(coordinator_module, "_BACKGROUND_INTERRUPTION_LEASE_SECONDS", 0.5)
    monkeypatch.setattr(coordinator_module, "_BACKGROUND_INTERRUPTION_HEARTBEAT_SECONDS", 0.01)

    async def scenario() -> None:
        store = _ClosableSessionStore()
        app = CayuApp(session_store=store, enable_logging=False, owned_resources=(store,))
        identity = SessionIdentity(provider_name="fake", model="fake-model")
        for session_id, parent, metadata in (
            ("parent", None, {}),
            ("child", "parent", {"subagent": {"mode": "background"}}),
        ):
            await store.create(
                RunRequest(
                    agent_name="r",
                    session_id=session_id,
                    parent_session_id=parent,
                    messages=[Message.text("user", "t")],
                    metadata=metadata,
                ),
                identity=identity,
            )
        coordinator = app._background_interruption_coordinator
        interrupting, release = asyncio.Event(), asyncio.Event()
        writes: list[bool] = []

        async def interrupt_child(request):
            interrupting.set()
            try:
                await asyncio.sleep(10)  # still interrupting when the drain times out
            except asyncio.CancelledError:
                # A durable interruption write that finishes after cancellation.
                await release.wait()
                await asyncio.sleep(0.05)
                writes.append(store.closed)
                raise
            yield Event(
                type=EventType.SESSION_INTERRUPTED,
                session_id=request.session_id,
                agent_name="r",
            )

        discard_queue = coordinator._discard_background_interruption_queue
        cancelled_drain: list[bool] = []

        def discard_then_cancel_the_drain() -> None:
            discard_queue()
            if not cancelled_drain:
                # The drain's caller cancels it while the cleanup yields.
                cancelled_drain.append(True)
                drain_task.cancel()

        monkeypatch.setattr(coordinator, "_interrupt_session", interrupt_child)
        app._session_finalization.schedule_background_interruption_cascade(
            parent_session_id="parent",
            interrupt_payload={
                "reason": "stop",
                "metadata": {},
                "requested_by": None,
                "interruption_type": "operator_requested",
            },
            create_if_missing=True,
        )
        await asyncio.wait_for(interrupting.wait(), 5)
        monkeypatch.setattr(
            coordinator, "_discard_background_interruption_queue", discard_then_cancel_the_drain
        )
        drain_task = asyncio.create_task(app.drain_background_interruptions(timeout_s=0.3))
        try:
            with pytest.raises(asyncio.CancelledError):
                await drain_task
            assert cancelled_drain and drain_task.cancelled()
            first = await app.aclose(timeout_s=0.3)
            assert not first.settled and not store.closed
        finally:
            release.set()
        assert (await app.aclose(timeout_s=5)).settled and store.closed
        assert writes == [False]

    asyncio.run(scenario())


def test_a_failed_claim_release_is_not_replaced_by_its_heartbeat_cancelling_the_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.core.test_provider_operation_offline_recovery import (
        _CompletionWinsCancellationProvider,
        _stage_offline_operation,
    )

    from cayu.agents import AgentSpec
    from cayu.runtime import _provider_operation_cancellation_owner as owner_module
    from cayu.sessions.requests import InterruptSessionRequest

    monkeypatch.setattr(
        owner_module, "_PROVIDER_OPERATION_CANCELLATION_CLAIM_HEARTBEAT_SECONDS", 0.01
    )

    class ReleaseFailed(RuntimeError):
        pass

    class RenewalFailed(RuntimeError):
        pass

    class FailingClaimWritesStore(_ClosableSessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.armed = False
            self.renewing = asyncio.Event()
            self.fail_renewal = asyncio.Event()

        async def promote_model_completion_stage(self, *args, **kwargs):
            promoted = await super().promote_model_completion_stage(*args, **kwargs)
            self.armed = True
            return promoted

        async def publish_checkpoint_and_events_with_store_time(self, session_id, **kwargs):
            transform = kwargs["checkpoint_transform"].__name__
            if self.armed and transform == "renew_claim":
                # A renewal already dispatched when the release fails.
                self.renewing.set()
                await self.fail_renewal.wait()
                raise RenewalFailed("renewal write failed")
            if self.armed and transform == "release_claim":
                await self.renewing.wait()
                asyncio.get_running_loop().call_later(0.05, self.fail_renewal.set)
                raise ReleaseFailed("release write failed")
            return await super().publish_checkpoint_and_events_with_store_time(session_id, **kwargs)

    async def scenario() -> None:
        store = FailingClaimWritesStore()
        provider = _CompletionWinsCancellationProvider()
        await _stage_offline_operation(store, session_id="won-completion", provider=provider)
        app = CayuApp(session_store=store, enable_logging=False, owned_resources=(store,))
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        failure: BaseException | None = None
        try:
            async for _ in app.interrupt_session(
                InterruptSessionRequest(session_id="won-completion", reason="stop")
            ):
                pass
        except BaseException as error:
            failure = error
        current = asyncio.current_task()
        assert current is not None and current.cancelling() == 0
        assert store.fail_renewal.is_set()
        assert failure is not None and not isinstance(failure, asyncio.CancelledError)
        chain: list[BaseException] = []
        error: BaseException | None = failure
        while error is not None and error not in chain:
            chain.append(error)
            error = error.__cause__ or error.__context__
        release_error = next(error for error in chain if isinstance(error, ReleaseFailed))
        assert any("RenewalFailed" in note for note in getattr(release_error, "__notes__", ()))
        assert (await app.aclose(timeout_s=5)).settled and store.closed

    asyncio.run(scenario())


def test_cancelling_a_failed_claim_release_while_it_stops_the_heartbeat_stays_a_cancellation() -> (
    None
):
    from types import SimpleNamespace

    class FailingReleaseStore(_ClosableSessionStore):
        async def publish_checkpoint_and_events_with_store_time(self, session_id, **kwargs):
            raise RuntimeError("release write failed")

    async def scenario() -> None:
        app = CayuApp(session_store=FailingReleaseStore(), enable_logging=False)
        control = _register_orphaned_heartbeat(app)
        heartbeat = control.task
        assert heartbeat is not None
        heartbeat.cancel()
        stopping, renewal_done = asyncio.Event(), asyncio.Event()

        async def slow_renewal() -> None:
            await control.stop.wait()
            stopping.set()
            await renewal_done.wait()

        control.task = asyncio.create_task(slow_renewal())
        await asyncio.gather(heartbeat, return_exceptions=True)
        owner = app._model_step_executor._provider_operation_cancellation
        release = asyncio.create_task(
            owner.release_claim(
                session=SimpleNamespace(id="session", run_epoch=1),  # ty: ignore[invalid-argument-type]
                claim=SimpleNamespace(claim_id="orphaned-claim"),  # ty: ignore[invalid-argument-type]
            )
        )
        await asyncio.wait_for(stopping.wait(), 5)
        # The caller cancels the release while it waits for the heartbeat.
        release.cancel()
        with pytest.raises(asyncio.CancelledError):
            await release
        assert release.cancelled()
        renewal_done.set()
        await control.task

    asyncio.run(scenario())
