"""HTTP server lifespan contracts."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
pytest.importorskip("httpx")

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.server.server_test_support import _LOCAL_SERVER_CONFIG

from cayu.applications import CayuApp
from cayu.budgets.base import InMemoryBudgetStore
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.observability.events import InMemoryEventSink
from cayu.runtime._event_projection import public_event_id
from cayu.server import (
    DashboardConfig,
    OpenAccess,
    ServerConfig,
    ServerLifecycleConfig,
    create_server,
    mount_cayu,
)
from cayu.sessions.event_delivery import PersistedEventSideEffectStatus
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.recovery import RecoveryExecutionRequest, RecoveryPlan
from cayu.sessions.requests import RunRequest


class FailOnceEventSink(InMemoryEventSink):
    def __init__(self) -> None:
        super().__init__()
        self.failures_remaining = 1

    async def emit(self, event: Event) -> None:
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("sink temporarily unavailable")
        await super().emit(event)


class NeverReturningEventSink(InMemoryEventSink):
    def __init__(self) -> None:
        super().__init__()
        self.cancelled = False

    async def emit(self, event: Event) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            self.cancelled = True


def test_mount_cayu_closes_the_app_under_one_shutdown_deadline() -> None:
    server = FastAPI()
    cayu_app = CayuApp()
    calls: list[tuple[str, float | None]] = []
    resume_calls = []

    async def resume_pending_interruption_cascades(*, interrupting_inactive_for_seconds):
        resume_calls.append(interrupting_inactive_for_seconds)
        return 0

    def drain(name, *, takes_s=0.0):
        async def record(*, timeout_s):
            calls.append((name, timeout_s))
            await asyncio.sleep(takes_s)
            return True

        return record

    async def close_runtime_timing():
        calls.append(("runtime_timing", None))

    # The first drain uses part of the deadline; later ones only get the rest.
    cayu_app.drain_background_interruptions = drain("drain_background_interruptions", takes_s=0.3)
    for name in (
        "drain_recovery_cleanups",
        "drain_provider_operation_cancellations",
        "drain_environment_cleanups",
        "close_idle_environment_resources",
        "drain_knowledge_publications",
        "drain_collaboration_requests",
        "drain_session_exports",
    ):
        setattr(cayu_app, name, drain(name))
    cayu_app.close_runtime_timing = close_runtime_timing
    cayu_app.resume_pending_interruption_cascades = resume_pending_interruption_cascades
    mount_cayu(
        server,
        cayu_app,
        path="/cayu",
        dashboard=False,
        access=OpenAccess(),
        interruption_shutdown_grace_seconds=2.5,
        knowledge_publication_shutdown_grace_seconds=1.5,
    )

    with TestClient(server):
        pass

    names = [name for name, _ in calls]
    assert names[:5] == [
        "drain_background_interruptions",
        "drain_recovery_cleanups",
        "drain_provider_operation_cancellations",
        "drain_environment_cleanups",
        "close_idle_environment_resources",
    ]
    assert set(names[5:8]) == {
        "drain_knowledge_publications",
        "drain_collaboration_requests",
        "drain_session_exports",
    }
    # Timing delivery stops after the other shutdown owners have drained.
    assert names[8:] == ["runtime_timing"]
    # Every drain shares the one 4.0s deadline instead of a fresh grace each.
    assert all(0 < timeout <= 4.0 for _, timeout in calls if timeout is not None)
    assert all(timeout <= 3.7 for _, timeout in calls[1:] if timeout is not None)
    assert cayu_app.lifecycle_state == "closed"
    outcome = cayu_app.shutdown_outcome
    assert outcome is not None and outcome.settled and outcome.timeout_seconds == 4.0
    assert len(resume_calls) == 1
    assert type(resume_calls[0]) is int
    assert resume_calls[0] >= 0


@pytest.mark.parametrize("host", ["mount_cayu", "create_server"])
def test_a_stalled_model_policy_stop_is_bounded_by_the_shutdown_deadline(host, caplog) -> None:
    cayu_app = CayuApp()
    stops: list[str] = []

    async def stalled_stop_model_policy():
        stops.append("stop")
        await asyncio.sleep(15)

    cayu_app.stop_model_policy = stalled_stop_model_policy  # ty: ignore[invalid-assignment]
    lifecycle = ServerLifecycleConfig(
        interruption_shutdown_grace_seconds=0.2,
        knowledge_publication_shutdown_grace_seconds=0.2,
    )
    if host == "mount_cayu":
        server = FastAPI()
        mount_cayu(
            server,
            cayu_app,
            path="/cayu",
            dashboard=False,
            access=OpenAccess(),
            interruption_shutdown_grace_seconds=0.2,
            knowledge_publication_shutdown_grace_seconds=0.2,
        )
    else:
        server = create_server(
            cayu_app,
            config=ServerConfig.local_development(
                dashboard=DashboardConfig(enabled=False), lifecycle=lifecycle
            ),
        )

    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="cayu.server"), TestClient(server):
        pass
    # The stalled stop no longer holds the host past its deadline.
    assert time.monotonic() - started < 10
    assert stops == ["stop"]
    outcome = cayu_app.shutdown_outcome
    assert outcome is not None and outcome.status == "incomplete"
    step = outcome.step("model_policy")
    assert step is not None and step.reason == "overran_budget"
    assert any("Model policy workers did not settle" in message for message in caplog.messages)


def test_mounted_shutdown_stops_model_policy_when_work_outlasts_the_deadline() -> None:
    cayu_app = CayuApp()
    stops: list[str] = []

    async def stop_model_policy():
        stops.append("stop")

    cayu_app.stop_model_policy = stop_model_policy  # ty: ignore[invalid-assignment]

    @asynccontextmanager
    async def host_lifespan(_server):
        # The host keeps one counted operation running past Cayu's shutdown.
        release = asyncio.Event()
        holder = asyncio.create_task(cayu_app._run_worker_step(release.wait))
        try:
            yield
        finally:
            release.set()
            await holder

    server = FastAPI(lifespan=host_lifespan)
    mount_cayu(
        server,
        cayu_app,
        path="/cayu",
        dashboard=False,
        access=OpenAccess(),
        interruption_shutdown_grace_seconds=0.1,
        knowledge_publication_shutdown_grace_seconds=0.1,
    )

    with TestClient(server):
        pass

    outcome = cayu_app.shutdown_outcome
    assert outcome is not None and outcome.status == "incomplete"
    assert outcome.open_operations == 1
    step = outcome.step("model_policy")
    assert step is not None and step.status == "settled"
    assert stops == ["stop"]


@pytest.mark.parametrize("host", ["mount_cayu", "create_server"])
def test_server_shutdown_stops_a_real_model_policy(host) -> None:
    from tests.core.test_model_policy_runtime import Channel, make_app

    from cayu.model_policy import InMemoryModelPolicyStore

    cayu_app, controller, _provider = make_app(InMemoryModelPolicyStore(), Channel())
    if host == "mount_cayu":
        server = FastAPI()
        mount_cayu(server, cayu_app, path="/cayu", dashboard=False, access=OpenAccess())
    else:
        server = create_server(
            cayu_app,
            config=ServerConfig.local_development(dashboard=DashboardConfig(enabled=False)),
        )

    with TestClient(server):
        assert controller.status == "ready"

    # The host started the workers and aclose() stopped them.
    assert controller.status == "stopped"
    assert cayu_app.model_policy is not None and not cayu_app.model_policy._tasks
    outcome = cayu_app.shutdown_outcome
    assert outcome is not None and outcome.settled
    step = outcome.step("model_policy")
    assert step is not None and step.status == "settled"


def test_every_shutdown_step_has_a_server_log_subject() -> None:
    from cayu.server import _SHUTDOWN_STEP_LOG_SUBJECTS

    outcome = asyncio.run(CayuApp(owned_resources=(_ClosingResource(),)).aclose())
    assert {step.subsystem for step in outcome.steps} == set(_SHUTDOWN_STEP_LOG_SUBJECTS)


class _ClosingResource:
    async def close(self) -> None:
        return None


def test_a_closed_app_cannot_start_another_server_lifespan() -> None:
    server = FastAPI()
    cayu_app = CayuApp()
    mount_cayu(server, cayu_app, path="/cayu", dashboard=False, access=OpenAccess())

    with TestClient(server):
        pass

    with (
        pytest.raises(RuntimeError, match="already shut down"),
        TestClient(server),
    ):
        pass


def test_mount_cayu_reports_unresolved_provider_cancellation_owners(caplog) -> None:
    server = FastAPI()
    cayu_app = CayuApp()

    async def unresolved_provider_cancellations(*, timeout_s):
        assert 0 < timeout_s <= 20.0
        return False

    cayu_app.drain_provider_operation_cancellations = unresolved_provider_cancellations
    cayu_app.provider_operation_cancellation_status = lambda: SimpleNamespace(
        active_owners=3, admission_rejections=0
    )
    mount_cayu(server, cayu_app, path="/cayu", dashboard=False, access=OpenAccess())

    with caplog.at_level(logging.WARNING, logger="cayu.server"), TestClient(server):
        pass

    assert (
        "3 provider-operation cancellation owner(s) remained unresolved after the "
        "20.000s shutdown grace period."
    ) in caplog.messages


def test_mount_cayu_drains_cascades_when_startup_recovery_fails() -> None:
    server = FastAPI()
    cayu_app = CayuApp()
    calls: list[str] = []

    async def resume_pending_interruption_cascades(*, interrupting_inactive_for_seconds):
        assert type(interrupting_inactive_for_seconds) is int
        assert interrupting_inactive_for_seconds >= 0
        calls.append("recover")
        raise RuntimeError("mounted recovery failed")

    async def drain_background_interruptions(*, timeout_s):
        assert 0 < timeout_s <= 20.0
        calls.append("drain")
        return True

    cayu_app.resume_pending_interruption_cascades = resume_pending_interruption_cascades
    cayu_app.drain_background_interruptions = drain_background_interruptions
    mount_cayu(server, cayu_app, path="/cayu", dashboard=False, access=OpenAccess())

    with (
        pytest.raises(RuntimeError, match="mounted recovery failed"),
        TestClient(server),
    ):
        pass

    assert calls == ["recover", "drain"]


def test_mount_cayu_recovers_backlog_past_poison_delivery(monkeypatch) -> None:
    class FailingBudgetStore(InMemoryBudgetStore):
        async def append_event(self, event: Event) -> None:
            if event.type == EventType.MODEL_COMPLETED:
                raise RuntimeError("budget unavailable")
            await super().append_event(event)

    async def prepare():
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="mounted_side_effect_backlog",
                messages=[Message.text("user", "go")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        poison = Event(type=EventType.MODEL_COMPLETED, session_id=session.id)
        healthy = Event(type="custom.healthy", session_id=session.id)
        await store.append_events(session.id, [poison, healthy])
        return store, poison, healthy

    store, poison, healthy = asyncio.run(prepare())
    sink = InMemoryEventSink()
    cayu_app = CayuApp(
        session_store=store,
        budget_store=FailingBudgetStore(),
        event_sinks=[sink],
        enable_logging=False,
    )
    server = FastAPI()
    monkeypatch.setattr(
        "cayu.server._PERSISTED_EVENT_SIDE_EFFECT_RECOVERY_BATCH_SIZE",
        1,
    )
    mount_cayu(server, cayu_app, path="/cayu", dashboard=False, access=OpenAccess())

    with TestClient(server):
        deliveries = asyncio.run(store.list_persisted_event_side_effect_deliveries())

    assert [event.id for event in sink.events] == [public_event_id(2)]
    assert [(delivery.event_id, delivery.status.value) for delivery in deliveries] == [
        (poison.id, "failed"),
        (healthy.id, "delivered"),
    ]


def test_create_server_startup_recovery_composes_user_lifespan() -> None:
    app = CayuApp()
    calls: list[str] = []
    requests = []

    @asynccontextmanager
    async def user_lifespan(server):
        calls.append("user_start")
        yield
        calls.append("user_stop")

    async def plan_recovery(request):
        calls.append("plan_recovery")
        requests.append(request)
        return RecoveryPlan(
            plan_id="startup-plan-one",
            created_at=datetime.now(UTC),
            request=request,
            items=(),
            inspected_session_count=0,
        )

    async def execute_recovery(request):
        calls.append("execute_recovery")
        assert isinstance(request, RecoveryExecutionRequest)
        assert request.execution_id == f"server-startup:{request.plan.plan_id}"
        assert request.max_concurrency == 8

    async def recover_event_side_effects(*, limit=1000):
        calls.append("recover_event_side_effects")
        assert limit == 1000
        return []

    async def drain_background_interruptions(*, timeout_s):
        calls.append("drain")
        # One 20.0s server deadline is shared by every shutdown step.
        assert 0 < timeout_s <= 20.0
        return True

    async def drain_provider_operation_cancellations(*, timeout_s):
        calls.append("provider_cancellation_drain")
        # One 20.0s server deadline is shared by every shutdown step.
        assert 0 < timeout_s <= 20.0
        return True

    async def resume_pending_interruption_cascades(*, interrupting_inactive_for_seconds):
        calls.append("resume_cascades")
        assert interrupting_inactive_for_seconds == 60
        return 0

    app.plan_recovery = plan_recovery
    app.execute_recovery = execute_recovery
    app.recover_persisted_event_side_effects = recover_event_side_effects
    app.drain_background_interruptions = drain_background_interruptions
    app.drain_provider_operation_cancellations = drain_provider_operation_cancellations
    app.resume_pending_interruption_cascades = resume_pending_interruption_cascades
    server = create_server(
        app,
        config=ServerConfig.local_development(
            lifecycle=ServerLifecycleConfig(
                startup_recovery_statuses={
                    SessionStatus.PENDING,
                    SessionStatus.RUNNING,
                    SessionStatus.INTERRUPTING,
                    SessionStatus.COMPLETED,
                    SessionStatus.FAILED,
                    SessionStatus.INTERRUPTED,
                },
                recovery_inactive_after_seconds=60,
            )
        ),
        fastapi_options={"lifespan": user_lifespan},
    )

    with TestClient(server):
        assert calls == [
            "user_start",
            "recover_event_side_effects",
            "plan_recovery",
            "execute_recovery",
            "resume_cascades",
        ]

    assert calls == [
        "user_start",
        "recover_event_side_effects",
        "plan_recovery",
        "execute_recovery",
        "resume_cascades",
        "drain",
        "provider_cancellation_drain",
        "user_stop",
    ]
    assert len(requests) == 1
    request = requests[0]
    assert request.selection.statuses == {
        SessionStatus.PENDING,
        SessionStatus.RUNNING,
        SessionStatus.INTERRUPTING,
        SessionStatus.COMPLETED,
        SessionStatus.FAILED,
        SessionStatus.INTERRUPTED,
    }
    assert request.selection.inactive_for_seconds == 60


def test_create_server_startup_recovery_consumes_every_cursor_page() -> None:
    app = CayuApp()
    requests = []
    server_is_ready = threading.Event()
    continuation_started = threading.Event()
    continuation_finished = threading.Event()

    async def plan_recovery(request):
        requests.append(request)
        if request.selection.cursor is None:
            next_cursor = "startup-page-2"
        elif request.selection.cursor == "startup-page-2":
            continuation_started.set()
            for _ in range(100):
                if server_is_ready.is_set():
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("Startup recovery continued before server readiness.")
            next_cursor = "startup-page-3"
        else:
            assert request.selection.cursor == "startup-page-3"
            continuation_finished.set()
            next_cursor = None
        return RecoveryPlan(
            plan_id=f"startup-plan-{len(requests)}",
            created_at=datetime.now(UTC),
            request=request,
            items=(),
            inspected_session_count=0,
            next_cursor=next_cursor,
        )

    async def execute_recovery(_request):
        return None

    app.plan_recovery = plan_recovery
    app.execute_recovery = execute_recovery
    server = create_server(
        app,
        config=ServerConfig.local_development(
            lifecycle=ServerLifecycleConfig(
                startup_recovery_statuses={SessionStatus.INTERRUPTED},
                recovery_inactive_after_seconds=60,
            )
        ),
    )

    with TestClient(server):
        server_is_ready.set()
        assert continuation_started.wait(timeout=10.0)
        assert continuation_finished.wait(timeout=2.0)

    assert [request.selection.cursor for request in requests] == [
        None,
        "startup-page-2",
        "startup-page-3",
    ]
    assert all(request.selection.statuses == {SessionStatus.INTERRUPTED} for request in requests)
    assert len({request.selection.inactive_for_seconds for request in requests}) == 1


def test_create_server_stops_incomplete_recovery_continuation_on_shutdown() -> None:
    app = CayuApp()
    continuation_started = threading.Event()
    continuation_cancelled = threading.Event()

    async def plan_recovery(request):
        if request.selection.cursor is None:
            return RecoveryPlan(
                plan_id="startup-plan-one",
                created_at=datetime.now(UTC),
                request=request,
                items=(),
                inspected_session_count=0,
                next_cursor="startup-page-2",
            )
        continuation_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            continuation_cancelled.set()
            raise

    async def execute_recovery(_request):
        return None

    app.plan_recovery = plan_recovery
    app.execute_recovery = execute_recovery
    server = create_server(
        app,
        config=ServerConfig.local_development(
            lifecycle=ServerLifecycleConfig(
                startup_recovery_statuses={SessionStatus.INTERRUPTED},
            )
        ),
    )

    with TestClient(server):
        assert continuation_started.wait(timeout=10.0)

    assert continuation_cancelled.wait(timeout=2.0)


def test_create_server_background_startup_recovery_rejects_cursor_cycles(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = CayuApp()
    cursors = []
    continuation_finished = threading.Event()

    async def plan_recovery(request):
        cursors.append(request.selection.cursor)
        if request.selection.cursor is not None:
            continuation_finished.set()
        return RecoveryPlan(
            plan_id=f"startup-plan-{len(cursors)}",
            created_at=datetime.now(UTC),
            request=request,
            items=(),
            inspected_session_count=0,
            next_cursor="repeated-startup-cursor",
        )

    async def execute_recovery(_request):
        return None

    app.plan_recovery = plan_recovery
    app.execute_recovery = execute_recovery
    server = create_server(
        app,
        config=ServerConfig.local_development(
            lifecycle=ServerLifecycleConfig(
                startup_recovery_statuses={SessionStatus.INTERRUPTED},
            )
        ),
    )

    with caplog.at_level(logging.ERROR, logger="cayu.server"), TestClient(server):
        assert continuation_finished.wait(timeout=2.0)
        deadline = time.monotonic() + 2.0
        while (
            "startup recovery returned a repeated cursor" not in caplog.text
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)

    assert cursors == [None, "repeated-startup-cursor"]
    assert "startup recovery returned a repeated cursor" in caplog.text


def test_create_server_drains_persisted_event_side_effect_backlog() -> None:
    async def prepare():
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="server_side_effect_backlog",
                messages=[Message.text("user", "go")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        events = [
            Event(type="custom.backlog", session_id=session.id, payload={"index": index})
            for index in range(1001)
        ]
        await store.append_events(session.id, events)
        return store, events

    store, events = asyncio.run(prepare())
    sink = InMemoryEventSink()
    app = CayuApp(session_store=store, event_sinks=[sink], enable_logging=False)

    with TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG)):
        assert [event.id for event in sink.events] == [
            public_event_id(index) for index in range(1, len(events) + 1)
        ]


@pytest.mark.parametrize("adapter", ["create_server", "mount_cayu"])
def test_server_startup_bounds_non_returning_side_effect_recovery(
    adapter: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def prepare():
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=f"bounded_startup_{adapter}",
                messages=[Message.text("user", "go")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        event = Event(type="custom.pending", session_id=session.id)
        await store.append_event(session.id, event)
        return store, event

    store, event = asyncio.run(prepare())
    sink = NeverReturningEventSink()
    app = CayuApp(session_store=store, event_sinks=[sink], enable_logging=False)
    if adapter == "create_server":
        server = create_server(
            app,
            config=ServerConfig.local_development(
                lifecycle=ServerLifecycleConfig(event_side_effect_startup_timeout_seconds=0.01)
            ),
        )
        health_path = "/api/health"
    else:
        server = FastAPI()
        mount_cayu(
            server,
            app,
            path="/cayu",
            dashboard=False,
            access=OpenAccess(),
            event_side_effect_startup_timeout_seconds=0.01,
        )
        health_path = "/cayu/api/health"

    started = time.monotonic()
    with (
        caplog.at_level(logging.WARNING, logger="cayu.server"),
        TestClient(server) as client,
    ):
        assert client.get(health_path).json() == {"ok": True}
    assert time.monotonic() - started < 2.0

    delivery = asyncio.run(
        store.get_persisted_event_side_effect_delivery(
            session_id=event.session_id,
            event_id=event.id,
        )
    )
    assert sink.cancelled
    assert delivery is not None
    assert delivery.status is PersistedEventSideEffectStatus.LEASED
    assert "startup recovery exceeded" in caplog.text


def test_create_server_retries_recovery_timeout_error(monkeypatch, caplog) -> None:
    app = CayuApp()
    attempts = 0
    monkeypatch.setattr("cayu.server._PERSISTED_EVENT_SIDE_EFFECT_RECOVERY_INTERVAL_SECONDS", 0.01)

    async def fail_recovery(*, limit):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("private session store timeout")
        return []

    app.recover_persisted_event_side_effects = fail_recovery
    server = create_server(app, config=_LOCAL_SERVER_CONFIG)

    async def wait_for_retry():
        status = server.state.cayu_event_side_effect_recovery
        async with asyncio.timeout(5):
            while status.sweep_successes == 0:
                await asyncio.sleep(0.01)
        assert status.sweep_failures == 1
        assert status.consecutive_failures == 0
        assert status.last_error_at is not None
        assert "private session store timeout" not in status.model_dump_json()

    with TestClient(server) as client:
        assert client.portal is not None
        client.portal.call(wait_for_retry)
    assert "startup sweep timed out" in caplog.text
    assert "private session store timeout" not in caplog.text


def test_create_server_retries_crash_claim_after_lease_expiry(monkeypatch) -> None:
    async def prepare():
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="server_side_effect_lease",
                messages=[Message.text("user", "go")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        event = Event(type="custom.leased", session_id=session.id)
        await store.append_event(session.id, event)
        claim = await store.claim_persisted_event_side_effect(lease_seconds=0.2)
        assert claim is not None
        return store, event

    store, _event = asyncio.run(prepare())
    sink = InMemoryEventSink()
    app = CayuApp(session_store=store, event_sinks=[sink], enable_logging=False)
    monkeypatch.setattr(
        "cayu.server._PERSISTED_EVENT_SIDE_EFFECT_RECOVERY_INTERVAL_SECONDS",
        0.01,
    )

    with TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG)):
        deadline = time.monotonic() + 1.0
        while not sink.events and time.monotonic() < deadline:
            time.sleep(0.01)

    assert [recovered.id for recovered in sink.events] == [public_event_id(1)]


def test_mount_cayu_retries_crash_claim_after_lease_expiry(monkeypatch) -> None:
    async def prepare():
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="mounted_side_effect_lease",
                messages=[Message.text("user", "go")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        event = Event(type="custom.leased", session_id=session.id)
        await store.append_event(session.id, event)
        claim = await store.claim_persisted_event_side_effect(lease_seconds=0.2)
        assert claim is not None
        return store, event

    store, _event = asyncio.run(prepare())
    sink = InMemoryEventSink()
    app = CayuApp(session_store=store, event_sinks=[sink], enable_logging=False)
    server = FastAPI()
    monkeypatch.setattr(
        "cayu.server._PERSISTED_EVENT_SIDE_EFFECT_RECOVERY_INTERVAL_SECONDS",
        0.01,
    )
    mount_cayu(server, app, path="/cayu", dashboard=False, access=OpenAccess())

    with TestClient(server):
        deadline = time.monotonic() + 1.0
        while not sink.events and time.monotonic() < deadline:
            time.sleep(0.01)

    assert [recovered.id for recovered in sink.events] == [public_event_id(1)]


def test_create_server_defers_transient_sink_retry_to_periodic_loop() -> None:
    async def prepare():
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="server_side_effect_transient_sink",
                messages=[Message.text("user", "go")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        event = Event(type="custom.transient", session_id=session.id)
        await store.append_event(session.id, event)
        return store

    store = asyncio.run(prepare())
    app = CayuApp(
        session_store=store,
        event_sinks=[FailOnceEventSink()],
        enable_logging=False,
    )

    with TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG)):
        deliveries = asyncio.run(store.list_persisted_event_side_effect_deliveries())

    assert [(delivery.status.value, delivery.attempts) for delivery in deliveries] == [
        ("failed", 1)
    ]
    assert deliveries[0].next_attempt_at is not None
    assert deliveries[0].next_attempt_at > deliveries[0].updated_at


def test_create_server_drains_cascades_when_startup_recovery_fails() -> None:
    app = CayuApp()
    calls: list[str] = []

    async def resume_pending_interruption_cascades(*, interrupting_inactive_for_seconds):
        assert type(interrupting_inactive_for_seconds) is int
        assert interrupting_inactive_for_seconds >= 0
        calls.append("recover")
        raise RuntimeError("recovery failed after scheduling work")

    async def drain_background_interruptions(*, timeout_s):
        # One 20.0s server deadline is shared by every shutdown step.
        assert 0 < timeout_s <= 20.0
        calls.append("drain")
        return True

    app.resume_pending_interruption_cascades = resume_pending_interruption_cascades
    app.drain_background_interruptions = drain_background_interruptions
    server = create_server(app, config=_LOCAL_SERVER_CONFIG)

    with (
        pytest.raises(RuntimeError, match="recovery failed after scheduling work"),
        TestClient(server),
    ):
        pass

    assert calls == ["recover", "drain"]


@pytest.mark.parametrize("value", [True, 0, -1, float("inf")])
def test_create_server_rejects_invalid_interruption_shutdown_grace(value) -> None:
    with pytest.raises(ValueError, match="interruption_shutdown_grace_seconds"):
        create_server(
            CayuApp(),
            config=ServerConfig.local_development(
                lifecycle=ServerLifecycleConfig(interruption_shutdown_grace_seconds=value)
            ),
        )


@pytest.mark.parametrize("value", [True, 0, -1, float("inf")])
def test_create_server_rejects_invalid_knowledge_publication_shutdown_grace(value) -> None:
    with pytest.raises(ValueError, match="knowledge_publication_shutdown_grace_seconds"):
        create_server(
            CayuApp(),
            config=ServerConfig.local_development(
                lifecycle=ServerLifecycleConfig(knowledge_publication_shutdown_grace_seconds=value)
            ),
        )


@pytest.mark.parametrize("value", [True, 0, -1, float("inf"), float("nan")])
@pytest.mark.parametrize("adapter", ["create_server", "mount_cayu"])
def test_server_rejects_invalid_event_side_effect_startup_timeout(
    adapter: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="event_side_effect_startup_timeout_seconds"):
        if adapter == "create_server":
            create_server(
                CayuApp(),
                config=ServerConfig.local_development(
                    lifecycle=ServerLifecycleConfig(
                        event_side_effect_startup_timeout_seconds=value  # type: ignore[arg-type]
                    )
                ),
            )
        else:
            mount_cayu(
                FastAPI(),
                CayuApp(),
                dashboard=False,
                access=OpenAccess(),
                event_side_effect_startup_timeout_seconds=value,  # type: ignore[arg-type]
            )


@pytest.mark.parametrize("value", [True, -1, 1.5])
def test_mount_cayu_rejects_invalid_interruption_recovery_inactivity(value) -> None:
    with pytest.raises(ValueError, match="interruption_recovery_inactive_after_seconds"):
        mount_cayu(
            FastAPI(),
            CayuApp(),
            dashboard=False,
            access=OpenAccess(),
            interruption_recovery_inactive_after_seconds=value,
        )
