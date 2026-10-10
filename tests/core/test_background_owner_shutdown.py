"""Shutdown waits for background work that outlives the call that started it."""

from __future__ import annotations

import asyncio
import contextlib
import faulthandler
import sys
import threading

import pytest
from tests.core.test_application_shutdown import _app, _consume, _GatedProvider
from tests.core.test_browser_control import operator_purpose
from tests.core.test_browser_control_authorization import Policy
from tests.core.test_event_watcher_leases import event_app
from tests.core.test_mcp import FakeMcpSession, _fake_toolset, _list_changed_initialize_result

from cayu import BrowserControlConfig
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.messages import Message
from cayu.observability.watchers import (
    EventWatcher,
    EventWatcherDeliveryStatus,
    InMemoryEventWatcherStore,
)
from cayu.runtime._compaction import automatic as _automatic_compaction
from cayu.runtime._compaction import explicit as _session_compaction
from cayu.runtime.application_lifecycle import ApplicationAdmissionsSealed
from cayu.sessions.base import InMemorySessionStore
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.requests import RunRequest


class _OwnedWatcherStore(InMemoryEventWatcherStore):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def test_shutdown_waits_for_a_watcher_delivery_retained_past_its_caller() -> None:
    async def scenario() -> None:
        store = _OwnedWatcherStore()
        prepared = await event_app(store)
        app = CayuApp(
            session_store=prepared.session_store,
            event_watcher_store=store,
            enable_logging=False,
            owned_resources=(store,),
        )
        started, release = threading.Event(), threading.Event()

        def handler(_context) -> None:
            started.set()
            assert release.wait(timeout=10)

        watcher = EventWatcher(
            name="retained", query=EventQuery(), handler=handler, lease_seconds=1
        )
        caller = asyncio.create_task(app.run_event_watchers([watcher]))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller

            # The synchronous handler still runs, renewing its lease; its
            # failure publication still needs the store.
            outcome = await app.aclose(timeout_s=0.5)
            assert outcome.status == "incomplete"
            step = outcome.step("event_watchers")
            assert step is not None and step.status == "incomplete"
            assert outcome.owned_resources == "retained" and not store.closed
        finally:
            release.set()

        settled = await app.aclose(timeout_s=10)
        assert settled.settled and store.closed
        state = await store.load_state(watcher.name)
        assert state.delivery_status is EventWatcherDeliveryStatus.FAILED

    asyncio.run(scenario())


def test_shutdown_stops_a_pending_mcp_notification_refresh() -> None:
    async def scenario() -> None:
        toolset = _fake_toolset(initialize_result=_list_changed_initialize_result())
        session = toolset.session
        assert isinstance(session, FakeMcpSession)
        app = CayuApp(enable_logging=False)
        app.register_agent(AgentSpec(name="first", model="fake-model"), mcp_toolsets=(toolset,))
        # A coalesced notification refresh is pending when the app closes.
        session.emit_tools_list_changed()
        assert (await app.aclose(timeout_s=5)).settled
        assert session.tools_list_changed_handler is None
        await asyncio.sleep(0.2)
        assert session.list_tools_calls == 0

    asyncio.run(scenario())


def test_shutdown_releases_a_refreshable_mcp_toolset_for_another_app() -> None:
    async def scenario() -> None:
        toolset = _fake_toolset(initialize_result=_list_changed_initialize_result())
        session = toolset.session
        assert isinstance(session, FakeMcpSession)
        first = CayuApp(enable_logging=False)
        first.register_agent(AgentSpec(name="first", model="fake-model"), mcp_toolsets=(toolset,))
        assert (await first.aclose(timeout_s=5)).settled

        second = CayuApp(enable_logging=False)
        second.register_agent(AgentSpec(name="second", model="fake-model"), mcp_toolsets=(toolset,))
        assert session.tools_list_changed_handler is not None
        assert (await second.aclose(timeout_s=5)).settled

    asyncio.run(scenario())


def test_shutdown_releases_a_static_mcp_registration() -> None:
    async def scenario() -> None:
        toolset = _fake_toolset(initialize_result=_list_changed_initialize_result())
        first = CayuApp(enable_logging=False)
        first.register_agent(AgentSpec(name="first", model="fake-model"), tools=toolset.tools)
        assert (await first.aclose(timeout_s=5)).settled

        second = CayuApp(enable_logging=False)
        second.register_agent(AgentSpec(name="second", model="fake-model"), mcp_toolsets=(toolset,))
        assert (await second.aclose(timeout_s=5)).settled

    asyncio.run(scenario())


def _browser_control_app(owned_resources: tuple[_OwnedWatcherStore, ...] = ()) -> CayuApp:
    return CayuApp(
        enable_logging=False,
        browser_control=BrowserControlConfig(
            purpose=operator_purpose(),
            policy=Policy(True),
            guest_endpoint="wss://operator.test/api/browser-control/guest",
        ),
        owned_resources=owned_resources,
    )


def test_shutdown_drains_browser_control_outside_a_server() -> None:
    async def scenario() -> None:
        app = _browser_control_app()
        runtime = app._browser_control_runtime
        assert runtime is not None
        outcome = await app.aclose(timeout_s=5)
        assert outcome.settled
        step = outcome.step("browser_control")
        assert step is not None and step.status == "settled"
        # Issuance stopped and publication sealed.
        assert runtime.service._closed
        assert runtime.coordinator._publisher._closing

    asyncio.run(scenario())


def test_unsettled_browser_control_keeps_owned_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        store = _OwnedWatcherStore()
        app = _browser_control_app(owned_resources=(store,))
        runtime = app._browser_control_runtime
        assert runtime is not None
        unsettled = True

        async def service_drain(*, timeout_s: float) -> bool:
            del timeout_s
            return not unsettled

        monkeypatch.setattr(runtime.service, "drain", service_drain)
        outcome = await app.aclose(timeout_s=5)
        step = outcome.step("browser_control")
        assert step is not None and step.status == "incomplete"
        assert outcome.owned_resources == "retained" and not store.closed
        # The publisher stays open while guest cleanup may still publish.
        assert not runtime.coordinator._publisher._closing

        unsettled = False
        assert (await app.aclose(timeout_s=5)).settled and store.closed

    asyncio.run(scenario())


def test_shutdown_waits_for_a_session_write_kept_past_its_bounded_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    monkeypatch.setattr(_session_compaction, "_SESSION_OPERATION_STORE_WAIT_TIMEOUT_SECONDS", 0.05)

    async def scenario() -> None:
        store = _OwnedWatcherStore()
        app = CayuApp(enable_logging=False, owned_resources=(store,))
        finished = asyncio.Event()

        async def opaque_write() -> None:
            # A store write that ignores cancellation until it completes.
            while not finished.is_set():
                with contextlib.suppress(asyncio.CancelledError):
                    await finished.wait()

        write = asyncio.create_task(opaque_write())
        try:
            outcome = await app._session_compaction._await_session_operation_store_task(write)
            assert outcome.timed_out and not write.done()

            closing = await app.aclose(timeout_s=0.3)
            step = closing.step("session_operations")
            assert step is not None and step.status == "incomplete"
            assert closing.owned_resources == "retained" and not store.closed
        finally:
            finished.set()
        assert (await app.aclose(timeout_s=5)).settled and store.closed

    asyncio.run(scenario())


class _SlowReleaseSessionStore(InMemorySessionStore):
    """Hold a run's execution-presence release, which can outlast the run."""

    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.releasing = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False

    async def _release_session_execution(self, expected):
        self.releasing.set()
        await self.release.wait()
        return await super()._release_session_execution(expected)

    async def close(self) -> None:
        self.closed = True


def test_a_presence_release_after_the_recovery_step_is_late_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.runtime import _session_execution_presence as presence_module

    # The run stops waiting for its release, which stays owned and pending.
    monkeypatch.setattr(presence_module, "_RELEASE_WAIT_SECONDS", 0.05)

    async def scenario() -> None:
        store = _SlowReleaseSessionStore()
        provider = _GatedProvider()
        provider.release.set()
        app = _app(provider, session_store=store, owned_resources=(store,))
        await _consume(app, "presence")
        await asyncio.wait_for(store.releasing.wait(), 5)

        async def settled_before_the_release(*, timeout_s: float) -> bool:
            # Stands in for a recovery step that finished before this release began.
            del timeout_s
            return True

        monkeypatch.setattr(app, "drain_recovery_cleanups", settled_before_the_release)
        try:
            outcome = await app.aclose(timeout_s=1)
            step = outcome.step("recovery_cleanups")
            assert step is not None and step.status == "incomplete"
            assert step.reason == "late_work"
            assert outcome.owned_resources == "retained" and not store.closed
        finally:
            store.release.set()
        assert (await app.aclose(timeout_s=5)).settled and store.closed

    asyncio.run(scenario())


def test_a_notification_while_shutdown_waits_for_a_run_does_not_spin(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A regression spins without yielding, so no in-loop timeout can fire; end
    # the process with a traceback instead of hanging the suite. Output capture
    # is disabled so the traceback reaches the terminal.
    with capsys.disabled():
        faulthandler.dump_traceback_later(60, exit=True, file=sys.stderr)
        try:
            asyncio.run(_notification_while_waiting_for_a_run())
        finally:
            faulthandler.cancel_dump_traceback_later()


async def _notification_while_waiting_for_a_run() -> None:
    toolset = _fake_toolset(initialize_result=_list_changed_initialize_result())
    session = toolset.session
    assert isinstance(session, FakeMcpSession)
    provider = _GatedProvider()
    app = _app(provider)
    app.register_agent(AgentSpec(name="tools", model="fake-model"), mcp_toolsets=(toolset,))
    run = asyncio.create_task(_consume(app, "in-flight"))
    await asyncio.wait_for(provider.started.wait(), 5)
    closing = asyncio.create_task(app.aclose(timeout_s=10))
    await asyncio.sleep(0.05)
    # Sealed, but still waiting for the run: the coalesced refresh fires
    # before the mcp_toolsets step can cancel it.
    session.emit_tools_list_changed()
    refresh = toolset._refresh_source._notification_refresh_task
    assert refresh is not None
    # The refresh fires while shutdown still waits for the run, and ends.
    await asyncio.wait_for(asyncio.gather(refresh, return_exceptions=True), 5)
    assert not run.done()
    provider.release.set()
    await run
    outcome = await asyncio.wait_for(closing, 10)
    assert outcome.settled
    assert session.tools_list_changed_handler is None
    assert session.list_tools_calls == 0


def test_a_closed_app_cannot_take_an_mcp_toolset_again() -> None:
    async def scenario() -> None:
        toolset = _fake_toolset(initialize_result=_list_changed_initialize_result())
        first = CayuApp(enable_logging=False)
        assert (await first.aclose(timeout_s=5)).settled
        with pytest.raises(ApplicationAdmissionsSealed):
            first.register_agent(
                AgentSpec(name="late", model="fake-model"), mcp_toolsets=(toolset,)
            )
        with pytest.raises(ApplicationAdmissionsSealed):
            first.register_agent(AgentSpec(name="late", model="fake-model"), tools=toolset.tools)
        assert first.list_agents() == ()

        second = CayuApp(enable_logging=False)
        second.register_agent(AgentSpec(name="second", model="fake-model"), mcp_toolsets=(toolset,))
        assert (await second.aclose(timeout_s=5)).settled

    asyncio.run(scenario())


def test_a_long_shutdown_deadline_still_drains_browser_control() -> None:
    async def scenario() -> None:
        app = _browser_control_app()
        assert await app.drain_browser_control(timeout_s=200) is True
        outcome = await _browser_control_app().aclose(timeout_s=200)
        assert outcome.settled

    asyncio.run(scenario())


def _protected_server(app: CayuApp):
    from tests.server.test_browser_control_server import transport

    from cayu.server import BasicAuth, ServerConfig, create_server

    return create_server(
        app,
        config=ServerConfig.protected(
            BasicAuth(username="operator", password="password"), browser_control=transport()
        ),
    )


def test_a_server_that_never_ran_leaves_browser_control_to_aclose() -> None:
    async def scenario() -> None:
        app = _browser_control_app()
        runtime = app._browser_control_runtime
        assert runtime is not None
        _protected_server(app)
        assert (await app.aclose(timeout_s=5)).settled
        assert runtime.service._closed and runtime.coordinator._publisher._closing

    asyncio.run(scenario())


def test_an_unsettled_server_browser_drain_keeps_owned_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        store = _OwnedWatcherStore()
        app = _browser_control_app(owned_resources=(store,))
        runtime = app._browser_control_runtime
        assert runtime is not None

        async def unsettled(**_options: object) -> bool:
            return False

        monkeypatch.setattr(runtime.service, "drain", unsettled)
        server = _protected_server(app)
        with pytest.raises(RuntimeError, match="Browser control shutdown remains unsettled"):
            async with server.router.lifespan_context(server):
                pass
        outcome = app.shutdown_outcome
        assert outcome is not None
        step = outcome.step("browser_control")
        assert step is not None and step.status == "incomplete"
        assert outcome.owned_resources == "retained" and not store.closed

    asyncio.run(scenario())


def test_retained_reconciliation_settles_before_provider_cancellation_is_sealed() -> None:
    async def scenario() -> None:
        app = CayuApp(enable_logging=False)
        release = asyncio.Event()
        sealed_when_settling: list[bool] = []

        async def reconciliation() -> None:
            await release.wait()
            # A late provider start would need an unsealed cancellation owner.
            status = app.provider_operation_cancellation_status()
            sealed_when_settling.append(status.admissions_sealed)

        app._model_step_executor._provider_operation_start._retain_reconciliation(
            asyncio.create_task(reconciliation())
        )
        closing = asyncio.create_task(app.aclose(timeout_s=5))
        await asyncio.sleep(0.05)
        release.set()
        assert (await closing).settled
        assert sealed_when_settling == [False]

    asyncio.run(scenario())


@pytest.mark.parametrize("server_drain", ["unsettled", "cancelled"])
def test_a_retry_drains_browser_control_after_the_server_drain_did_not_settle(
    monkeypatch: pytest.MonkeyPatch, server_drain: str
) -> None:
    async def scenario() -> None:
        store = _OwnedWatcherStore()
        app = _browser_control_app(owned_resources=(store,))
        runtime = app._browser_control_runtime
        assert runtime is not None
        real_drain = runtime.service.drain
        entered = asyncio.Event()

        async def server_side_drain(**_options: object) -> bool:
            entered.set()
            if server_drain == "cancelled":
                await asyncio.Event().wait()
            return False

        monkeypatch.setattr(runtime.service, "drain", server_side_drain)
        server = _protected_server(app)

        async def lifespan() -> None:
            async with server.router.lifespan_context(server):
                pass

        host = asyncio.create_task(lifespan())
        await asyncio.wait_for(entered.wait(), 5)
        if server_drain == "cancelled":
            host.cancel()
        with contextlib.suppress(RuntimeError, asyncio.CancelledError):
            await host
        first = app.shutdown_outcome
        assert first is not None and not first.settled and not store.closed

        # Guest work has settled since; a retry drains browser control itself.
        monkeypatch.setattr(runtime.service, "drain", real_drain)
        retried = await app.aclose(timeout_s=5)
        assert retried.settled and store.closed
        assert runtime.service._closed and runtime.coordinator._publisher._closing

    asyncio.run(scenario())


def test_shutdown_waits_for_a_model_step_write_kept_past_its_bounded_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.core.test_compaction_prefix_continuation import (
        PrefixSummarizer,
        context_request,
        policy_for,
    )

    from cayu.events import EventType

    monkeypatch.setattr(_automatic_compaction, "_CONTEXT_TERMINATION_PERSIST_TIMEOUT_S", 0.02)

    class OpaqueCheckpointStore(InMemorySessionStore):
        """Write the compaction checkpoint only once released, ignoring cancellation."""

        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.writing = asyncio.Event()
            self.release = asyncio.Event()
            self.closed = False

        async def publish_checkpoint_and_events(self, session_id, **kwargs):
            if not any(
                event.type == EventType.SESSION_CHECKPOINTED
                and event.payload.get("checkpoint") == "context_compaction"
                for event in kwargs["events"]
            ):
                return await super().publish_checkpoint_and_events(session_id, **kwargs)
            self.writing.set()
            while not self.release.is_set():
                with contextlib.suppress(asyncio.CancelledError):
                    await self.release.wait()
            return await super().publish_checkpoint_and_events(session_id, **kwargs)

        async def close(self) -> None:
            self.closed = True

    async def scenario() -> None:
        store = OpaqueCheckpointStore()
        app = CayuApp(session_store=store, enable_logging=False, owned_resources=(store,))
        app.register_provider(PrefixSummarizer(), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="synthetic"),
            context_policy=policy_for(PrefixSummarizer("cancelled")),
        )
        try:
            with contextlib.suppress(asyncio.CancelledError):
                async for _ in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="opaque-writer",
                        messages=(await context_request()).messages,
                    )
                ):
                    pass
            await asyncio.wait_for(store.writing.wait(), 5)
            # The run gave up waiting; the write it started is still running.
            outcome = await app.aclose(timeout_s=0.3)
            step = outcome.step("session_operations")
            assert step is not None and step.status == "incomplete"
            assert outcome.owned_resources == "retained" and not store.closed
        finally:
            store.release.set()
        # The write was cancelled with an unknown outcome when its wait timed
        # out; its late result is reconciled from durable state, not reported.
        assert (await app.aclose(timeout_s=5)).settled and store.closed

    asyncio.run(scenario())


def test_a_hung_watcher_does_not_starve_later_cleanup() -> None:
    async def scenario() -> None:
        store = InMemoryEventWatcherStore()
        prepared = await event_app(store)
        app = CayuApp(
            session_store=prepared.session_store, event_watcher_store=store, enable_logging=False
        )
        toolset = _fake_toolset(initialize_result=_list_changed_initialize_result())
        app.register_agent(AgentSpec(name="tools", model="fake-model"), mcp_toolsets=(toolset,))
        started, release = threading.Event(), threading.Event()

        def handler(_context) -> None:
            started.set()
            assert release.wait(timeout=10)

        watcher = EventWatcher(name="hung", query=EventQuery(), handler=handler, lease_seconds=1)
        caller = asyncio.create_task(app.run_event_watchers([watcher]))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            caller.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await caller
            outcome = await app.aclose(timeout_s=0.5)
            watchers = outcome.step("event_watchers")
            assert watchers is not None and watchers.status == "incomplete"
            for subsystem in ("environment_cleanups", "mcp_toolsets"):
                step = outcome.step(subsystem)
                assert step is not None and step.status == "settled", subsystem
            # The caller's toolset is free for another app.
            other = CayuApp(enable_logging=False)
            other.register_agent(
                AgentSpec(name="other", model="fake-model"), mcp_toolsets=(toolset,)
            )
            assert (await other.aclose(timeout_s=5)).settled
        finally:
            release.set()
        assert (await app.aclose(timeout_s=10)).settled

    asyncio.run(scenario())


def test_mcp_toolsets_stay_fenced_while_a_run_is_in_flight() -> None:
    async def scenario() -> None:
        toolset = _fake_toolset(initialize_result=_list_changed_initialize_result())
        session = toolset.session
        assert isinstance(session, FakeMcpSession)
        provider = _GatedProvider()
        app = _app(provider)
        app.register_agent(AgentSpec(name="tools", model="fake-model"), mcp_toolsets=(toolset,))
        run = asyncio.create_task(_consume(app, "in-flight"))
        await asyncio.wait_for(provider.started.wait(), 5)

        outcome = await app.aclose(timeout_s=0.2)
        step = outcome.step("mcp_toolsets")
        assert step is not None and step.status == "incomplete"
        # The running operation keeps this app's list-changed fence.
        assert session.tools_list_changed_handler is not None

        provider.release.set()
        await run
        assert (await app.aclose(timeout_s=5)).settled
        assert session.tools_list_changed_handler is None

    asyncio.run(scenario())


def test_a_concurrent_aclose_waits_for_the_server_browser_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        store = _OwnedWatcherStore()
        app = _browser_control_app(owned_resources=(store,))
        runtime = app._browser_control_runtime
        assert runtime is not None
        real_drain = runtime.service.drain
        entered, gate = asyncio.Event(), asyncio.Event()
        calls = 0

        async def gated_drain(**options: float) -> bool:
            nonlocal calls
            calls += 1
            entered.set()
            await gate.wait()
            return await real_drain(**options)

        monkeypatch.setattr(runtime.service, "drain", gated_drain)
        server = _protected_server(app)

        async def lifespan() -> None:
            async with server.router.lifespan_context(server):
                pass

        host = asyncio.create_task(lifespan())
        await asyncio.wait_for(entered.wait(), 5)
        # A user shutdown arrives while the server's browser drain runs.
        user = asyncio.create_task(app.aclose(timeout_s=10))
        await asyncio.sleep(0.05)
        # The user's shutdown waits for the server's drain instead of reporting
        # it early or closing resources under it.
        assert not user.done() and not store.closed
        # It waits for the server's drain rather than starting its own.
        assert calls == 1
        gate.set()
        await asyncio.wait_for(host, 10)
        outcome = await asyncio.wait_for(user, 10)
        assert outcome.settled and store.closed
        step = outcome.step("browser_control")
        assert step is not None and step.status == "settled"

    asyncio.run(scenario())


def test_a_hung_provider_reconciliation_does_not_starve_later_steps() -> None:
    async def scenario() -> None:
        app = CayuApp(enable_logging=False)
        release = asyncio.Event()

        async def hung_provider_start() -> None:
            await release.wait()

        app._model_step_executor._provider_operation_start._retain_reconciliation(
            asyncio.create_task(hung_provider_start())
        )
        try:
            outcome = await app.aclose(timeout_s=1)
            step = outcome.step("provider_reconciliations")
            assert step is not None and step.status == "incomplete"
            for subsystem in ("recovery_cleanups", "environment_cleanups", "event_watchers"):
                later = outcome.step(subsystem)
                assert later is not None and later.status == "settled", subsystem
        finally:
            release.set()
        assert (await app.aclose(timeout_s=5)).settled

    asyncio.run(scenario())


def test_mcp_toolsets_wait_for_a_short_operation_still_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        toolset = _fake_toolset(initialize_result=_list_changed_initialize_result())
        session = toolset.session
        assert isinstance(session, FakeMcpSession)
        app = CayuApp(enable_logging=False)
        app.register_agent(AgentSpec(name="tools", model="fake-model"), mcp_toolsets=(toolset,))
        background: list[asyncio.Task[None]] = []

        async def environment_cleanups(*, timeout_s: float) -> bool:
            del timeout_s

            async def short_step() -> None:
                await asyncio.sleep(0.1)

            # Counted work, allowed while closing, still running when the
            # final stage starts.
            background.append(asyncio.create_task(app._run_worker_step(short_step)))
            await asyncio.sleep(0)
            return True

        monkeypatch.setattr(app, "drain_environment_cleanups", environment_cleanups)
        outcome = await app.aclose(timeout_s=5)
        assert outcome.settled
        assert session.tools_list_changed_handler is None
        await asyncio.gather(*background)

    asyncio.run(scenario())


def test_mcp_toolsets_are_released_after_the_deadline_is_used_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        toolset = _fake_toolset(initialize_result=_list_changed_initialize_result())
        session = toolset.session
        assert isinstance(session, FakeMcpSession)
        app = CayuApp(enable_logging=False)
        app.register_agent(AgentSpec(name="tools", model="fake-model"), mcp_toolsets=(toolset,))

        async def slow_environment_cleanups(*, timeout_s: float) -> bool:
            await asyncio.sleep(timeout_s + 0.1)
            return True

        monkeypatch.setattr(app, "drain_environment_cleanups", slow_environment_cleanups)
        outcome = await app.aclose(timeout_s=0.3)
        assert not outcome.settled
        step = outcome.step("mcp_toolsets")
        assert step is not None and step.status == "settled"
        assert session.tools_list_changed_handler is None
        other = CayuApp(enable_logging=False)
        other.register_agent(AgentSpec(name="other", model="fake-model"), mcp_toolsets=(toolset,))
        assert (await other.aclose(timeout_s=5)).settled

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "drain", ["drain_browser_control", "drain_event_watchers", "drain_session_operations"]
)
def test_public_drains_validate_and_settle_when_idle(drain: str) -> None:
    async def scenario() -> None:
        app = _browser_control_app()
        for invalid in (0, -1.0, float("inf"), float("nan"), True, "1"):
            with pytest.raises(ValueError, match="finite positive"):
                await getattr(app, drain)(timeout_s=invalid)
        assert await getattr(app, drain)(timeout_s=1) is True

    asyncio.run(scenario())


class _UnpublishableWatcherStore(_OwnedWatcherStore):
    """Settlement publication fails, as when the watcher database is unavailable."""

    async def mark_success(self, claim):
        raise ConnectionError("watcher database unavailable")

    async def mark_failure(self, claim, *, error, max_attempts):
        raise ConnectionError("watcher database unavailable")


async def _watcher_app(store: _OwnedWatcherStore) -> CayuApp:
    prepared = await event_app(store)
    return CayuApp(
        session_store=prepared.session_store,
        event_watcher_store=store,
        enable_logging=False,
        owned_resources=(store,),
    )


def _start_delivery(app: CayuApp, handler) -> asyncio.Task[None]:
    watcher = EventWatcher(name="late", query=EventQuery(), handler=handler, lease_seconds=1)
    caller = asyncio.create_task(app.run_event_watchers([watcher]))
    return caller


@pytest.mark.parametrize("fails", ["during_aclose", "before_aclose"])
def test_a_watcher_settlement_that_failed_after_its_caller_left_is_reported(fails: str) -> None:
    async def scenario() -> None:
        store = _UnpublishableWatcherStore()
        app = await _watcher_app(store)
        started, release, finished = threading.Event(), threading.Event(), threading.Event()

        def handler(_context) -> None:
            started.set()
            assert release.wait(timeout=10)
            finished.set()

        caller = _start_delivery(app, handler)
        try:
            assert await asyncio.to_thread(started.wait, 5)
            caller.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await caller
            if fails == "during_aclose":
                assert not (await app.aclose(timeout_s=0.3)).settled
            release.set()
            assert await asyncio.to_thread(finished.wait, 5)
            if fails == "before_aclose":
                # Let the delivery fail to publish before shutdown starts.
                async with asyncio.timeout(5):
                    while app._event_watcher_supervisor.active("late"):
                        await asyncio.sleep(0.01)
        finally:
            release.set()
        failed = await app.aclose(timeout_s=5)
        step = failed.step("event_watchers")
        assert step is not None and step.status == "failed"
        assert step.failure_type == "RetainedWorkFailed"
        assert failed.owned_resources == "retained" and not store.closed
        # Reported once: a retry settles.
        assert (await app.aclose(timeout_s=5)).settled and store.closed

    asyncio.run(scenario())


def test_a_recorded_handler_failure_is_not_a_shutdown_failure() -> None:
    async def scenario() -> None:
        store = _OwnedWatcherStore()
        app = await _watcher_app(store)
        started, release = threading.Event(), threading.Event()

        def handler(_context) -> None:
            started.set()
            assert release.wait(timeout=10)
            raise ValueError("handler failed")

        caller = _start_delivery(app, handler)
        try:
            assert await asyncio.to_thread(started.wait, 5)
            caller.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await caller
        finally:
            release.set()
        assert (await app.aclose(timeout_s=10)).settled and store.closed
        state = await store.load_state("late")
        assert state.delivery_status is EventWatcherDeliveryStatus.FAILED

    asyncio.run(scenario())


async def _failing(message: str) -> None:
    raise ConnectionError(message)


@pytest.mark.parametrize("owner", ["final_session_accounting", "provider_reconciliation"])
def test_retained_work_that_failed_after_its_caller_left_is_reported_once(owner: str) -> None:
    async def scenario() -> None:
        store = _OwnedWatcherStore()
        app = CayuApp(enable_logging=False, owned_resources=(store,))
        task = asyncio.create_task(_failing("late accounting failed"))
        if owner == "final_session_accounting":
            app._session_compaction._track_detached_session_operation_task(
                task, report_failure=True
            )
            subsystem = "session_operations"
        else:
            app._model_step_executor._provider_operation_start._retain_reconciliation(task)
            subsystem = "provider_reconciliations"
        with contextlib.suppress(ConnectionError):
            await task

        failed = await app.aclose(timeout_s=5)
        step = failed.step(subsystem)
        assert step is not None and step.status == "failed"
        assert step.failure_type == "RetainedWorkFailed"
        assert failed.owned_resources == "retained" and not store.closed
        assert (await app.aclose(timeout_s=5)).settled and store.closed

    asyncio.run(scenario())


@pytest.mark.parametrize("owner", ["model_step_write", "session_write"])
def test_a_cancelled_kept_write_that_fails_is_not_reported(owner: str) -> None:
    async def scenario() -> None:
        store = _OwnedWatcherStore()
        app = CayuApp(enable_logging=False, owned_resources=(store,))
        task = asyncio.create_task(_failing("late write rejected"))
        if owner == "model_step_write":
            app._automatic_compaction._retain_detached_task(task)
        else:
            app._session_compaction._track_detached_session_operation_task(
                task, report_failure=False
            )
        with contextlib.suppress(ConnectionError):
            await task
        # Its caller already treated the outcome as unknown.
        assert (await app.aclose(timeout_s=5)).settled and store.closed

    asyncio.run(scenario())


def test_a_failed_late_cancellation_is_reported_unless_admission_was_sealed() -> None:
    from cayu.runtime._provider_operation_cancellation_owner import (
        _attach_provider_operation_cleanup_failure,
    )
    from cayu.runtime._provider_operation_start_owner import _late_cancellation_failure
    from cayu.runtime.provider_operation_cancellation import (
        ProviderOperationCancellationAdmissionsSealed,
    )

    assert _late_cancellation_failure(RuntimeError("start")) is None
    cancel_failed = ConnectionError("provider cancel failed")
    failure = RuntimeError("start")
    _attach_provider_operation_cleanup_failure(failure, cancel_failed)
    assert _late_cancellation_failure(failure) is cancel_failed
    # Shutdown already reports a sealed admission as an unowned cancellation.
    sealed = RuntimeError("start")
    _attach_provider_operation_cleanup_failure(
        sealed, ProviderOperationCancellationAdmissionsSealed("sealed")
    )
    assert _late_cancellation_failure(sealed) is None


def test_a_settlement_failure_its_caller_received_is_not_reported_again() -> None:
    async def scenario() -> None:
        store = _UnpublishableWatcherStore()
        app = await _watcher_app(store)
        watcher = EventWatcher(
            name="observed", query=EventQuery(), handler=lambda _context: None, lease_seconds=1
        )
        results = await app.run_event_watchers([watcher])
        assert any(
            delivery.status is EventWatcherDeliveryStatus.PUBLICATION_FAILED
            for result in results
            for delivery in result.deliveries
        )
        assert (await app.aclose(timeout_s=5)).settled and store.closed

    asyncio.run(scenario())


def test_a_failure_after_its_step_ran_is_late_work_then_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.core.test_provider_operations import _LateSuccessfulStartAdapter

    import cayu.runtime._provider_operation_cancellation_owner as cancellation_owner
    import cayu.runtime._provider_operation_start_owner as start_owner

    monkeypatch.setattr(start_owner, "_PROVIDER_OPERATION_START_SETTLEMENT_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr(
        cancellation_owner, "_PROVIDER_OPERATION_START_CLEANUP_TIMEOUT_SECONDS", 0.05
    )

    class SlowFailingCancel(_LateSuccessfulStartAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.cancel_release = asyncio.Event()

        async def cancel(self, state):
            self.late_cancel_observed.set()
            while not self.cancel_release.is_set():
                with contextlib.suppress(asyncio.CancelledError):
                    await self.cancel_release.wait()
            raise TimeoutError("provider cancel failed late")

    async def scenario() -> None:
        adapter = SlowFailingCancel()
        app = await _late_provider_start(adapter)
        # The reconciliation finishes; the lifecycle still owns the cancel.
        assert await app._model_step_executor.drain_provider_reconciliations(timeout_s=5)
        drain_cancellations = app.drain_provider_operation_cancellations

        async def cancel_fails_meanwhile(*, timeout_s: float) -> bool:
            # The cancel fails during a later step, after provider_reconciliations ran.
            asyncio.get_running_loop().call_later(0.05, adapter.cancel_release.set)
            return await drain_cancellations(timeout_s=timeout_s)

        monkeypatch.setattr(app, "drain_provider_operation_cancellations", cancel_fails_meanwhile)
        first = await app.aclose(timeout_s=5)
        step = first.step("provider_reconciliations")
        assert step is not None and step.status == "incomplete"
        assert step.reason == "late_work"
        failed = await app.aclose(timeout_s=5)
        step = failed.step("provider_reconciliations")
        assert step is not None and step.status == "failed"
        assert (await app.aclose(timeout_s=5)).settled

    asyncio.run(scenario())


async def _late_provider_start(
    adapter, owned_resources: tuple[_OwnedWatcherStore, ...] = ()
) -> CayuApp:
    """Cancel a run while its provider start is pending, then let the start land."""

    from tests.core.test_provider_operations import _collect_run_events, _ReconnectableProvider

    provider = _ReconnectableProvider(background=True)
    provider.adapter = adapter
    app = CayuApp(enable_logging=False, owned_resources=owned_resources)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    run = asyncio.create_task(
        _collect_run_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="late-start",
                messages=[Message.text("user", "hello")],
            ),
        )
    )
    await asyncio.wait_for(adapter.start_entered.wait(), 10)
    run.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await run
    adapter.start_release.set()
    await asyncio.wait_for(adapter.late_cancel_observed.wait(), 5)
    return app


@pytest.mark.parametrize("cancel", ["fails", "succeeds"])
def test_a_late_start_whose_cancellation_fails_is_reported_once(
    monkeypatch: pytest.MonkeyPatch, cancel: str
) -> None:
    from tests.core.test_provider_operations import (
        _LateFailingCancelAdapter,
        _LateSuccessfulStartAdapter,
    )

    import cayu.runtime._provider_operation_start_owner as start_owner

    monkeypatch.setattr(start_owner, "_PROVIDER_OPERATION_START_SETTLEMENT_TIMEOUT_SECONDS", 0.0)

    async def scenario() -> None:
        adapter = (
            _LateFailingCancelAdapter() if cancel == "fails" else _LateSuccessfulStartAdapter()
        )
        app = await _late_provider_start(adapter)
        outcome = await app.aclose(timeout_s=5)
        step = outcome.step("provider_reconciliations")
        assert step is not None
        if cancel == "succeeds":
            assert outcome.settled
            return
        assert step.status == "failed" and step.failure_type == "RetainedWorkFailed"
        assert (await app.aclose(timeout_s=5)).settled

    asyncio.run(scenario())


def test_a_late_cancellation_that_fails_after_its_bounded_wait_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.core.test_provider_operations import _LateSuccessfulStartAdapter

    import cayu.runtime._provider_operation_cancellation_owner as cancellation_owner
    import cayu.runtime._provider_operation_start_owner as start_owner

    monkeypatch.setattr(start_owner, "_PROVIDER_OPERATION_START_SETTLEMENT_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr(
        cancellation_owner, "_PROVIDER_OPERATION_START_CLEANUP_TIMEOUT_SECONDS", 0.05
    )

    class SlowFailingCancel(_LateSuccessfulStartAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.cancel_release = asyncio.Event()
            self.cancel_done = asyncio.Event()

        async def cancel(self, state):
            self.late_cancel_observed.set()
            try:
                # Ignores local cancellation, outliving both bounded waits.
                while not self.cancel_release.is_set():
                    with contextlib.suppress(asyncio.CancelledError):
                        await self.cancel_release.wait()
                raise TimeoutError("provider cancel failed late")
            finally:
                self.cancel_done.set()

    async def scenario() -> None:
        adapter = SlowFailingCancel()
        app = await _late_provider_start(adapter)
        try:
            # The reconciliation gives up on the cancellation and finishes.
            assert await app._model_step_executor.drain_provider_reconciliations(timeout_s=5)
        finally:
            adapter.cancel_release.set()
        await asyncio.wait_for(adapter.cancel_done.wait(), 5)
        outcome = await app.aclose(timeout_s=5)
        assert not outcome.settled
        step = outcome.step("provider_reconciliations")
        assert step is not None and step.status == "failed"
        assert (await app.aclose(timeout_s=5)).settled

    asyncio.run(scenario())


def test_a_timed_out_session_write_that_then_fails_is_not_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    monkeypatch.setattr(_session_compaction, "_SESSION_OPERATION_STORE_WAIT_TIMEOUT_SECONDS", 0.05)

    async def scenario() -> None:
        store = _OwnedWatcherStore()
        app = CayuApp(enable_logging=False, owned_resources=(store,))
        release = asyncio.Event()

        async def opaque_write() -> None:
            while not release.is_set():
                with contextlib.suppress(asyncio.CancelledError):
                    await release.wait()
            raise ConnectionError("late write rejected")

        write = asyncio.create_task(opaque_write())
        try:
            outcome = await app._session_compaction._await_session_operation_store_task(write)
            assert outcome.timed_out
        finally:
            release.set()
        with contextlib.suppress(ConnectionError):
            await write
        # Its caller already received the timeout and reconciles the write.
        assert (await app.aclose(timeout_s=5)).settled and store.closed

    asyncio.run(scenario())


def test_superseded_final_accounting_is_not_reported() -> None:
    from cayu.runtime._compaction.explicit import SessionCompactionAttemptSuperseded

    async def superseded() -> None:
        raise SessionCompactionAttemptSuperseded("a recovering owner took over")

    async def scenario() -> None:
        store = _OwnedWatcherStore()
        app = CayuApp(enable_logging=False, owned_resources=(store,))
        task = asyncio.create_task(superseded())
        app._session_compaction._track_detached_session_operation_task(task, report_failure=True)
        with contextlib.suppress(SessionCompactionAttemptSuperseded):
            await task
        # The recovering owner owns that outcome now.
        assert (await app.aclose(timeout_s=5)).settled and store.closed

    asyncio.run(scenario())


def test_a_late_start_whose_cancellation_admission_was_sealed_is_reported_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.core.test_provider_operations import (
        _collect_run_events,
        _LateSuccessfulStartAdapter,
        _ReconnectableProvider,
    )

    import cayu.runtime._provider_operation_start_owner as start_owner

    monkeypatch.setattr(start_owner, "_PROVIDER_OPERATION_START_SETTLEMENT_TIMEOUT_SECONDS", 0.0)

    async def scenario() -> None:
        adapter = _LateSuccessfulStartAdapter()
        provider = _ReconnectableProvider(background=True)
        provider.adapter = adapter
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        run = asyncio.create_task(
            _collect_run_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sealed-late-start",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
        await asyncio.wait_for(adapter.start_entered.wait(), 10)
        run.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run
        # Provider cancellation is sealed before the late start lands.
        assert await app.drain_provider_operation_cancellations(timeout_s=1)
        adapter.start_release.set()
        assert await app._model_step_executor.drain_provider_reconciliations(timeout_s=5)

        first = await app.aclose(timeout_s=5)
        # Reported once, as an unowned cancellation, not also as a reconciliation failure.
        cancellations = first.step("provider_operation_cancellations")
        assert cancellations is not None and cancellations.reason == "unowned_cancellations"
        reconciliations = first.step("provider_reconciliations")
        assert reconciliations is not None and reconciliations.status == "settled"
        assert (await app.aclose(timeout_s=5)).settled

    asyncio.run(scenario())


def test_cancelling_the_public_session_drain_keeps_an_unreported_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.core.test_provider_operations import _LateFailingCancelAdapter

    import cayu.runtime._provider_operation_start_owner as start_owner

    monkeypatch.setattr(start_owner, "_PROVIDER_OPERATION_START_SETTLEMENT_TIMEOUT_SECONDS", 0.0)

    async def scenario() -> None:
        store = _OwnedWatcherStore()
        app = await _late_provider_start(_LateFailingCancelAdapter(), owned_resources=(store,))
        start = app._model_step_executor._provider_operation_start
        async with asyncio.timeout(5):
            while not start.reconciliation_failures.pending:
                await asyncio.sleep(0.01)
        # Another retained write keeps the combined drain waiting.
        release = asyncio.Event()
        app._automatic_compaction._retain_detached_task(asyncio.create_task(release.wait()))

        drain = asyncio.create_task(app.drain_session_operations(timeout_s=10))
        await asyncio.sleep(0.05)
        assert not drain.done()
        drain.cancel()
        with pytest.raises(asyncio.CancelledError):
            await drain
        assert drain.cancelled()
        release.set()

        # The cancelled drain consumed nothing: shutdown still reports it once.
        failed = await app.aclose(timeout_s=5)
        step = failed.step("provider_reconciliations")
        assert step is not None and step.status == "failed"
        assert failed.owned_resources == "retained" and not store.closed
        assert (await app.aclose(timeout_s=5)).settled and store.closed

    asyncio.run(scenario())


def test_cancelling_the_public_session_drain_keeps_a_final_accounting_failure() -> None:
    async def scenario() -> None:
        store = _OwnedWatcherStore()
        app = CayuApp(enable_logging=False, owned_resources=(store,))
        accounting = asyncio.create_task(_failing("final accounting failed"))
        app._session_compaction._track_detached_session_operation_task(
            accounting, report_failure=True
        )
        with contextlib.suppress(ConnectionError):
            await accounting
        # Another retained write keeps the combined drain waiting.
        release = asyncio.Event()
        app._automatic_compaction._retain_detached_task(asyncio.create_task(release.wait()))

        drain = asyncio.create_task(app.drain_session_operations(timeout_s=10))
        await asyncio.sleep(0.05)
        assert not drain.done()
        drain.cancel()
        with pytest.raises(asyncio.CancelledError):
            await drain
        release.set()

        failed = await app.aclose(timeout_s=5)
        step = failed.step("session_operations")
        assert step is not None and step.status == "failed"
        assert failed.owned_resources == "retained" and not store.closed
        assert (await app.aclose(timeout_s=5)).settled and store.closed

    asyncio.run(scenario())


class _FirstWatcherUnpublishableStore(_OwnedWatcherStore):
    """Watchers named in ``unpublishable`` cannot publish their settlement."""

    def __init__(self, unpublishable: frozenset[str] = frozenset({"first"})) -> None:
        super().__init__()
        self.unpublishable = unpublishable

    async def mark_success(self, claim):
        if claim.watcher_name in self.unpublishable:
            raise ConnectionError("watcher database unavailable")
        return await super().mark_success(claim)

    async def mark_failure(self, claim, *, error, max_attempts):
        if claim.watcher_name in self.unpublishable:
            raise ConnectionError("watcher database unavailable")
        return await super().mark_failure(claim, error=error, max_attempts=max_attempts)


@pytest.mark.parametrize("call", ["cancelled", "returned"])
def test_a_collected_settlement_failure_is_reported_only_if_its_caller_never_got_it(
    call: str,
) -> None:
    async def scenario() -> None:
        store = _FirstWatcherUnpublishableStore()
        app = await _watcher_app(store)
        started, release = threading.Event(), threading.Event()

        def blocking(_context) -> None:
            started.set()
            assert release.wait(timeout=10)

        settled = EventWatcher(
            name="settled", query=EventQuery(), handler=lambda _context: None, lease_seconds=1
        )
        first = EventWatcher(
            name="first", query=EventQuery(), handler=lambda _context: None, lease_seconds=1
        )
        second = EventWatcher(
            name="second",
            query=EventQuery(),
            handler=blocking if call == "cancelled" else (lambda _context: None),
            lease_seconds=1,
        )
        caller = asyncio.create_task(app.run_event_watchers([settled, first, second]))
        try:
            if call == "cancelled":
                assert await asyncio.to_thread(started.wait, 5)
                caller.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await caller
            else:
                results = await caller
                assert any(
                    delivery.status is EventWatcherDeliveryStatus.PUBLICATION_FAILED
                    for result in results
                    for delivery in result.deliveries
                )
        finally:
            release.set()
        if call == "returned":
            # The caller received the failure, so shutdown does not report it again.
            assert (await app.aclose(timeout_s=5)).settled and store.closed
            return
        # Only the collected, unsettled delivery is reported, once.
        with pytest.raises(RuntimeError, match=r"\(publication_failed\)\."):
            await app.drain_event_watchers(timeout_s=5)
        assert (await app.aclose(timeout_s=5)).settled and store.closed

    asyncio.run(scenario())


def test_a_collected_and_the_running_settlement_failure_are_each_reported_once() -> None:
    async def scenario() -> None:
        store = _FirstWatcherUnpublishableStore(frozenset({"first", "second"}))
        app = await _watcher_app(store)
        started, release = threading.Event(), threading.Event()

        def blocking(_context) -> None:
            started.set()
            assert release.wait(timeout=10)

        first = EventWatcher(
            name="first", query=EventQuery(), handler=lambda _context: None, lease_seconds=1
        )
        second = EventWatcher(name="second", query=EventQuery(), handler=blocking, lease_seconds=1)
        caller = asyncio.create_task(app.run_event_watchers([first, second]))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller
        finally:
            release.set()
        # One from the collected result, one from the running delivery's callback.
        with pytest.raises(RuntimeError, match=r"\(publication_failed x2\)\."):
            await app.drain_event_watchers(timeout_s=5)
        assert (await app.aclose(timeout_s=5)).settled and store.closed

    asyncio.run(scenario())
