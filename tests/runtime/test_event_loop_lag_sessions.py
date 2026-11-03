"""A worker's event-loop lag recorded on the sessions it slowed, and read back by the CLI."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

import pytest

from cayu import CayuApp
from cayu.cli import main
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.observability import event_loop_lag
from cayu.observability.event_loop_lag import EventLoopLagMonitor
from cayu.runtime._durable_worker_loop import DurableWorkerMetrics, worker_event_loop_lag_monitor
from cayu.runtime._event_loop_lag_sessions import SessionEventLoopLagRecorder
from cayu.sessions._event_loop_lag import event_loop_lag_summary, event_loop_lag_text
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest
from cayu.storage import SQLiteSessionStore, SQLiteTaskStore
from cayu.tasks import TaskCreate


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class _Sessions:
    def __init__(self, *ids: str) -> None:
        self.ids = set(ids)

    def __call__(self) -> tuple[str, ...]:
        return tuple(sorted(self.ids))


def _recorder(sessions: _Sessions, emitted: list[Event], clock: _Clock, **kwargs):
    async def emit(event: Event) -> Event:
        emitted.append(event)
        return event

    return SessionEventLoopLagRecorder(
        active_session_ids=sessions,
        emit=kwargs.pop("emit", emit),
        monotonic=clock,
        **kwargs,
    )


def test_late_samples_are_charged_to_the_sessions_running_then() -> None:
    sessions, emitted, clock = _Sessions("a", "b"), [], _Clock()
    recorder = _recorder(sessions, emitted, clock)

    async def scenario() -> None:
        recorder.sample(0.1)  # on time
        recorder.sample(0.5)
        clock.now += 0.5
        sessions.ids = {"a", "c"}  # b ended; c started
        recorder.sample(0.9)
        clock.now += 30
        recorder.warned(0.9)
        await recorder.aclose()

    asyncio.run(scenario())

    assert [event.session_id for event in emitted] == ["a", "c"]
    first, second = emitted
    assert first.type == EventType.RUNTIME_EVENT_LOOP_LAGGED
    assert {key: value for key, value in first.payload.items() if key != "pid"} == {
        "lag_ms": 900,
        "max_lag_ms": 900,
        "lagging_samples": 2,
        "lagging_ms": 1000,
        "window_ms": 31_000,
        "sample_interval_ms": 500,
        "threshold_ms": 250,
        "sessions_running": 2,
    }
    assert (second.payload["lagging_samples"], second.payload["window_ms"]) == (1, 30_500)
    assert type(first.payload["pid"]) is int


def test_a_write_still_queued_keeps_counting_and_an_ended_session_gets_nothing() -> None:
    sessions, emitted, clock = _Sessions("a", "b"), [], _Clock()
    release = asyncio.Event()

    async def slow_emit(event: Event) -> Event:
        await release.wait()
        emitted.append(event)
        return event

    recorder = _recorder(sessions, emitted, clock, emit=slow_emit)

    async def scenario() -> None:
        recorder.sample(0.4)
        recorder.warned(0.4)
        await asyncio.sleep(0)
        # The first write waits; the next late samples keep counting.
        clock.now += 0.5
        recorder.sample(0.6)
        recorder.warned(0.6)
        sessions.ids = {"a"}  # b ends before its write runs
        release.set()
        await recorder.aclose()
        recorder.warned(0.6)
        await recorder.aclose()

    asyncio.run(scenario())

    assert [(event.session_id, event.payload["lagging_samples"]) for event in emitted] == [
        ("a", 1),
        ("a", 1),
    ]


def test_a_failed_write_is_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    sessions, clock = _Sessions("a", "b"), _Clock()

    async def fail(event: Event) -> Event:
        raise RuntimeError("store is down")

    recorder = _recorder(sessions, [], clock, emit=fail)

    async def scenario() -> None:
        recorder.sample(0.4)
        recorder.warned(0.4)
        await recorder.aclose()

    with caplog.at_level(logging.DEBUG, logger="cayu.runtime._event_loop_lag_sessions"):
        asyncio.run(scenario())

    levels = [record.levelno for record in caplog.records]
    assert levels == [logging.WARNING, logging.DEBUG]


def test_the_worker_monitor_feeds_metrics_and_the_recorder_and_warns_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(event_loop_lag, "_last_warning_at", None)
    metrics = DurableWorkerMetrics()
    sessions, emitted, clock = _Sessions("a"), [], _Clock()
    recorder = _recorder(sessions, emitted, clock, threshold_s=0.05, sample_interval_s=0.01)
    monitor = worker_event_loop_lag_monitor(
        metrics, active_sessions=lambda: len(sessions.ids), session_lag=recorder
    )
    # Sample fast and warn on one late sample.
    monitor._interval_s = 0.01
    monitor._warning_threshold_s = 0.05
    monitor._sustained_samples = 1

    async def scenario() -> None:
        monitor.start()
        await asyncio.sleep(0.03)
        time.sleep(0.1)
        await asyncio.sleep(0.03)
        monitor.stop()
        await recorder.aclose()

    asyncio.run(scenario())

    assert metrics.snapshot().event_loop_lag_samples == monitor.snapshot().samples
    (event,) = emitted
    assert event.session_id == "a"
    assert event.payload["max_lag_ms"] >= 50


def test_a_failing_warning_callback_is_dropped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(event_loop_lag, "_last_warning_at", None)
    calls = 0

    def fail(_lag: float) -> None:
        nonlocal calls
        calls += 1
        raise ValueError("boom")

    monitor = EventLoopLagMonitor(
        interval_s=0.01,
        warning_threshold_s=0.05,
        sustained_samples=1,
        warning_interval_s=0.01,
        on_warning=fail,
    )

    async def scenario() -> None:
        monitor.start()
        for _ in range(2):
            await asyncio.sleep(0.03)
            time.sleep(0.1)
        await asyncio.sleep(0.03)
        monitor.stop()

    with caplog.at_level("WARNING", logger="cayu.observability.event_loop_lag"):
        asyncio.run(scenario())
    assert calls == 1
    assert "An event-loop lag warning callback failed and was removed." in [
        record.getMessage() for record in caplog.records
    ]
    with pytest.raises(TypeError, match="on_warning"):
        EventLoopLagMonitor(on_warning=1)  # type: ignore[arg-type]


def test_only_an_app_gets_a_recorder_over_its_running_sessions() -> None:
    from cayu.runtime._event_loop_lag_sessions import app_session_event_loop_lag_recorder

    assert app_session_event_loop_lag_recorder(object()) is None
    recorder = app_session_event_loop_lag_recorder(CayuApp(enable_logging=False))
    assert isinstance(recorder, SessionEventLoopLagRecorder)
    # No session runs in a fresh app, so a late sample is charged to none.
    recorder.sample(1.0)
    assert recorder._sessions == {}


def test_workers_of_one_app_share_a_recorder_that_counts_each_interval_once() -> None:
    import gc
    import weakref

    from cayu.runtime._event_loop_lag_sessions import app_session_event_loop_lag_recorder

    app = CayuApp(enable_logging=False)
    assert app_session_event_loop_lag_recorder(app) is app_session_event_loop_lag_recorder(app)
    # The recorder lives with its app, not past it.
    released = weakref.ref(app)
    del app
    gc.collect()
    assert released() is None

    sessions, emitted, clock = _Sessions("a"), [], _Clock()
    recorder = _recorder(sessions, emitted, clock)

    async def scenario() -> None:
        # Two workers' monitors sample the same late loop each interval.
        for _ in range(10):
            clock.now += 0.5
            recorder.sample(0.6)
            clock.now += 0.01
            recorder.sample(0.6)
        recorder.warned(0.6)
        await recorder.aclose()
        # The other worker's warning finds nothing left to report.
        recorder.warned(0.6)
        await recorder.aclose()

    asyncio.run(scenario())

    assert [event.payload["lagging_samples"] for event in emitted] == [10]


def test_the_summary_adds_up_reports_and_says_nothing_without_them() -> None:
    assert event_loop_lag_summary([]) == {
        "reports": 0,
        "lagging_ms": 0,
        "max_lag_ms": None,
        "first_at": None,
        "last_at": None,
        "processes": 0,
        "truncated": False,
    }
    assert event_loop_lag_text(event_loop_lag_summary([])) is None
    assert event_loop_lag_text(None) is None


@pytest.fixture
def lagged_store(tmp_path: Path) -> tuple[Path, str]:
    """A session two lag reports were written to through the app, and a task that owns it."""

    database = tmp_path / "cayu.db"

    async def seed() -> str:
        store = SQLiteSessionStore(database)
        app = CayuApp(session_store=store, enable_logging=False)
        try:
            await store.create(
                RunRequest(
                    agent_name="hunter", session_id="lagged", messages=[Message.text("user", "x")]
                ),
                identity=SessionIdentity(provider_name="fake", model="m"),
            )
            clock = _Clock()
            recorder = SessionEventLoopLagRecorder(
                active_session_ids=lambda: ("lagged",),
                emit=app._event_writer.emit,
                monotonic=clock,
            )
            for lag in (0.7, 1.2):
                for _ in range(120):
                    clock.now += 0.5
                    recorder.sample(lag)
                recorder.warned(lag)
                await recorder.aclose()
        finally:
            await store.close()
        tasks = SQLiteTaskStore(database)
        try:
            task = await tasks.create_task(TaskCreate(type="hunt", session_id="lagged"))
        finally:
            await tasks.close()
        return task.id

    return database, asyncio.run(seed())


def test_session_and_task_show_report_the_lag(lagged_store, capsys) -> None:
    database, task_id = lagged_store

    assert main(["session", "show", "lagged", "--sqlite", str(database), "--json"]) == 0
    lag = json.loads(capsys.readouterr().out)["event_loop_lag"]
    assert {key: lag[key] for key in ("reports", "lagging_ms", "max_lag_ms", "processes")} == {
        "reports": 2,
        "lagging_ms": 120_000,
        "max_lag_ms": 1200,
        "processes": 1,
    }

    assert main(["session", "show", "lagged", "--sqlite", str(database), "--table"]) == 0
    text = capsys.readouterr().out
    assert "late for 2m00s while this session ran, up to 1.20 s (2 reports" in text

    assert main(["task", "show", task_id, "--sqlite", str(database), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["event_loop_lag"] == lag

    assert main(["task", "show", task_id, "--sqlite", str(database), "--table"]) == 0
    text = capsys.readouterr().out
    assert "Event loop" in text and "up to 1.20 s" in text

    assert (
        main(
            [
                "session",
                "events",
                "lagged",
                "--sqlite",
                str(database),
                "--type",
                "runtime.event_loop.lagged",
                "--json",
            ]
        )
        == 0
    )
    assert len(json.loads(capsys.readouterr().out)["events"]) == 2


def test_a_session_without_lag_reports_zero(lagged_store, capsys) -> None:
    database, _ = lagged_store

    async def seed() -> None:
        store = SQLiteSessionStore(database)
        try:
            await store.create(
                RunRequest(
                    agent_name="hunter", session_id="calm", messages=[Message.text("user", "x")]
                ),
                identity=SessionIdentity(provider_name="fake", model="m"),
            )
        finally:
            await store.close()

    asyncio.run(seed())
    assert main(["session", "show", "calm", "--sqlite", str(database), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["event_loop_lag"]["reports"] == 0
    assert main(["session", "show", "calm", "--sqlite", str(database), "--table"]) == 0
    assert "event loop was late" not in capsys.readouterr().out


@pytest.fixture
def agent(lagged_store, monkeypatch):
    """The lagged store served as a Cloud Agent, and a signed-in CLI."""

    from fastapi.testclient import TestClient
    from tests.cli.test_session_cli_cloud import _FakeCloud

    from cayu.server import ServerConfig, create_server
    from cayu.server.auth import BasicAuth

    database, task_id = lagged_store
    monkeypatch.setattr("cayu.cli.cloud.cloud_client", lambda **_: _FakeCloud())
    server = create_server(
        CayuApp(
            session_store=SQLiteSessionStore(database),
            task_store=SQLiteTaskStore(database),
            enable_logging=False,
        ),
        config=ServerConfig.protected(BasicAuth(username="operator", password="s3cret-password")),
    )
    with TestClient(server, base_url="http://127.0.0.1") as client:
        monkeypatch.setattr(client, "close", lambda: None)
        monkeypatch.setattr("cayu.cli._cloud_agent_api.httpx.Client", lambda **_: client)
        yield database, task_id


def _show(capsys, *argv: str) -> dict:
    assert main([*argv, "--json"]) == 0, capsys.readouterr()
    return json.loads(capsys.readouterr().out)


def test_a_cloud_agent_reports_the_same_lag(agent, capsys) -> None:
    database, task_id = agent
    remote = "--application", "bug-hunter-agent"

    local = _show(capsys, "session", "show", "lagged", "--sqlite", str(database))
    shown = _show(capsys, "session", "show", "lagged", *remote)
    assert shown["event_loop_lag"] == local["event_loop_lag"]
    assert shown["event_loop_lag"]["reports"] == 2

    task = _show(capsys, "task", "show", task_id, *remote)
    assert task["event_loop_lag"] == local["event_loop_lag"]


def test_an_older_agent_reports_no_lag(agent, capsys, monkeypatch) -> None:
    from cayu.cli._agent_session_reader import AgentSessionReader
    from cayu.cli._cloud_agent_api import CayuApiError

    async def older(self, query):
        raise CayuApiError("request_rejected", "unknown event type", status_code=422)

    monkeypatch.setattr(AgentSessionReader, "query_events", older)
    _, task_id = agent
    remote = "--application", "bug-hunter-agent"

    assert _show(capsys, "session", "show", "lagged", *remote)["event_loop_lag"] is None
    assert _show(capsys, "task", "show", task_id, *remote)["event_loop_lag"] is None


def test_task_show_still_shows_the_task_when_the_lag_read_fails_for_now(
    agent, capsys, monkeypatch
) -> None:
    from cayu.cli._agent_session_reader import AgentSessionReader
    from cayu.cli._cloud_agent_api import CayuApiError

    async def unavailable(self, query):
        raise CayuApiError("server_error", "Agent answered HTTP 503", status_code=503)

    monkeypatch.setattr(AgentSessionReader, "query_events", unavailable)
    _, task_id = agent

    shown = _show(capsys, "task", "show", task_id, "--application", "bug-hunter-agent")
    assert shown["task"]["id"] == task_id
    assert shown["event_loop_lag"] is None
