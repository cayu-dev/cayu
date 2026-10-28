"""Brokered coding sessions that wait in Cayu Cloud's queue for sandbox capacity."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from tests.sandbox_broker.fake_broker import FakeBroker

from cayu.environments import (
    EnvironmentFactoryOperation,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
)
from cayu.environments.lifecycle import (
    EnvironmentLifecycleDeadlineExceeded,
    EnvironmentLifecycleOperation,
    EnvironmentLifecyclePhase,
    EnvironmentLifecyclePolicy,
    EnvironmentLifecycleProgress,
    EnvironmentLifecycleProgressStatus,
    RuntimeEnvironmentLifecycleProgressReporter,
    _reset_environment_lifecycle_progress_reporter,
    _set_environment_lifecycle_progress_reporter,
)
from cayu.sandbox_broker import (
    BrokeredSandboxCodingEnvironmentFactory,
    SandboxBrokerClient,
    SandboxBrokerError,
    SandboxBrokerUnavailableError,
    SandboxQueuedRequest,
    SandboxRecord,
    report_sandbox_queue_position,
)
from cayu.sandbox_broker import coding as coding_module
from cayu.sandbox_broker.client import MAX_QUEUE_SECONDS
from cayu.sandbox_broker.coding import BrokeredCodingBackendOptions
from cayu.workspaces import LocalWorkspace

_PROVISIONING = EnvironmentLifecyclePhase.FACTORY_PROVISIONING
_ADVANCED = EnvironmentLifecycleProgressStatus.ADVANCED


def _request(**options: Any) -> EnvironmentFactoryRequest:
    return EnvironmentFactoryRequest(
        session_id="coding", agent_name="coder", environment_name="sandbox", **options
    )


def _factory(
    broker: FakeBroker, tmp_path: Path, client: SandboxBrokerClient | None = None, **overrides
) -> BrokeredSandboxCodingEnvironmentFactory:
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    (source / "calc.py").write_bytes(b"def add(a, b):\n    return a + b\n")
    options: dict[str, Any] = {
        "broker": client or broker.client(),
        "source_workspace": LocalWorkspace(source, workspace_id="project"),
        "egress": "none",
        "git_baseline": False,
    }
    options.update(overrides)
    return BrokeredSandboxCodingEnvironmentFactory(**options)


def _reporter(
    published: list[EnvironmentLifecycleProgress], **policy: Any
) -> RuntimeEnvironmentLifecycleProgressReporter:
    async def publish(progress: EnvironmentLifecycleProgress) -> None:
        published.append(progress)

    return RuntimeEnvironmentLifecycleProgressReporter(
        operation=EnvironmentLifecycleOperation.FACTORY,
        policy=EnvironmentLifecyclePolicy(progress_min_interval_seconds=0.0, **policy),
        publish=publish,
    )


def _record_ready_waits(client: SandboxBrokerClient, monkeypatch) -> list[float]:
    """Record each ``wait_until_ready`` timeout, polling fast so tests don't wait a second."""

    timeouts: list[float] = []
    wait = client.wait_until_ready

    async def recording(sandbox_id: str, *, timeout_s: float, **_: Any) -> SandboxRecord:
        timeouts.append(timeout_s)
        return await wait(sandbox_id, timeout_s=timeout_s, poll_interval_s=0.01)

    monkeypatch.setattr(client, "wait_until_ready", recording)
    return timeouts


async def _occupy(broker: FakeBroker) -> tuple[SandboxBrokerClient, SandboxRecord]:
    client = broker.client()
    record = await client.create(
        idempotency_key="running", image="coding-python", egress_mode="none", lease_seconds=60
    )
    return client, await client.wait_until_ready(record.id, timeout_s=5, poll_interval_s=0)


async def _until(condition: Callable[[], bool]) -> None:
    for _ in range(500):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never held")


def _create_bodies(broker: FakeBroker) -> list[dict[str, Any]]:
    return [
        json.loads(request.content)
        for request in broker.requests
        if request.method == "POST" and request.url.path.endswith("/sandboxes")
    ]


@pytest.mark.parametrize("value", [-1, MAX_QUEUE_SECONDS + 1, True, 1.5, "60", None])
def test_queue_seconds_is_checked_against_clouds_maximum(tmp_path: Path, value: Any) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    with pytest.raises(ValueError, match="queue_seconds"):
        _factory(broker, tmp_path, queue_seconds=value)
    with pytest.raises(ValueError, match="queue_seconds"):
        BrokeredCodingBackendOptions(queue_seconds=value)
    assert _factory(broker, tmp_path).queue_seconds == 0
    assert _factory(broker, tmp_path, queue_seconds=MAX_QUEUE_SECONDS).queue_seconds == 3600


def test_without_queue_seconds_a_create_doesnt_queue(tmp_path: Path, monkeypatch) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario() -> None:
        client = broker.client()
        timeouts = _record_ready_waits(client, monkeypatch)
        result = await _factory(broker, tmp_path, client).create(_request())
        assert result.release is not None
        await result.release(EnvironmentFactoryReleaseAction.DISCARD)
        # The body older Runtimes sent, and the ready wait they used.
        assert "queue_seconds" not in _create_bodies(broker)[0]
        assert timeouts == [330.0]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("operation", "phase"),
    [
        (EnvironmentFactoryOperation.CREATE, _PROVISIONING),
        # A follow-up turn's create runs under the reconnect phase, and its deadline.
        (EnvironmentFactoryOperation.RECONNECT, EnvironmentLifecyclePhase.FACTORY_RECONNECT),
    ],
)
def test_a_queued_session_reports_its_position_and_starts_once_admitted(
    tmp_path: Path,
    monkeypatch,
    operation: EnvironmentFactoryOperation,
    phase: EnvironmentLifecyclePhase,
) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    broker.agent_limit = 1
    monkeypatch.setattr(coding_module, "_QUEUE_POLL_INTERVAL_SECONDS", 0.01)
    published: list[EnvironmentLifecycleProgress] = []

    async def scenario() -> None:
        occupier, running = await _occupy(broker)
        client = broker.client()
        timeouts = _record_ready_waits(client, monkeypatch)
        factory = _factory(broker, tmp_path, client, queue_seconds=600)
        reporter = _reporter(published)
        await reporter.report(phase, EnvironmentLifecycleProgressStatus.STARTED)
        token = _set_environment_lifecycle_progress_reporter(reporter)
        try:
            creating = asyncio.create_task(factory.create(_request(operation=operation)))
        finally:
            _reset_environment_lifecycle_progress_reporter(token)
        await _until(lambda: any(item.status is _ADVANCED for item in published))
        assert not creating.done()
        assert _create_bodies(broker)[-1]["queue_seconds"] == 600
        # The ready wait grows by the queue's: 600 plus the default 330.
        assert timeouts == [930.0]

        # Once the running sandbox ends, the queued one is admitted and the session starts.
        await occupier.destroy(running.id)
        broker.expire(running.id)
        result = await asyncio.wait_for(creating, 5)
        reports = [item for item in published if item.status is _ADVANCED]
        assert reports and all(item.phase is phase for item in reports)
        assert reports[0].queued_count == 1
        await asyncio.sleep(0.05)
        # Nothing more is reported once the sandbox is out of the queue.
        assert [item for item in published if item.status is _ADVANCED] == reports
        assert result.release is not None
        await result.release(EnvironmentFactoryReleaseAction.DISCARD)
        await occupier.aclose()

    asyncio.run(scenario())
    assert broker.live() == []


def test_a_queued_create_without_a_reporter_still_waits(tmp_path: Path, monkeypatch) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    broker.agent_limit = 1

    async def scenario() -> None:
        occupier, running = await _occupy(broker)
        client = broker.client()
        _record_ready_waits(client, monkeypatch)
        creating = asyncio.create_task(
            _factory(broker, tmp_path, client, queue_seconds=60).create(_request())
        )
        await asyncio.sleep(0.1)
        assert not creating.done()
        await occupier.destroy(running.id)
        broker.expire(running.id)
        result = await asyncio.wait_for(creating, 5)
        assert result.release is not None
        await result.release(EnvironmentFactoryReleaseAction.DISCARD)
        # Without lifecycle progress nothing reads the queue.
        assert ("GET", "/capacity") not in broker.calls
        await occupier.aclose()

    asyncio.run(scenario())


def test_a_queue_that_outlasts_the_wait_destroys_the_sandbox(tmp_path: Path, monkeypatch) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    broker.agent_limit = 1

    async def scenario() -> None:
        occupier, _ = await _occupy(broker)
        client = broker.client()
        timeouts = _record_ready_waits(client, monkeypatch)
        factory = _factory(broker, tmp_path, client, queue_seconds=1, ready_timeout_s=0.1)
        with pytest.raises(SandboxBrokerUnavailableError, match="within 1.1 seconds"):
            await factory.create(_request())
        assert timeouts == [1.1]
        await occupier.aclose()

    asyncio.run(scenario())
    # Only the sandbox that held the place is left.
    assert [sandbox.id for sandbox in broker.live()] == ["sbx_0001"]


def test_a_lifecycle_deadline_ends_a_queued_wait(tmp_path: Path, monkeypatch) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    broker.agent_limit = 1
    monkeypatch.setattr(coding_module, "_QUEUE_POLL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(coding_module, "_QUEUE_HEARTBEAT_SECONDS", 0.02)
    published: list[EnvironmentLifecycleProgress] = []

    async def scenario() -> None:
        occupier, _ = await _occupy(broker)
        client = broker.client()
        _record_ready_waits(client, monkeypatch)
        reporter = _reporter(published, phase_timeout_seconds={_PROVISIONING: 0.2})
        await reporter.report(_PROVISIONING, EnvironmentLifecycleProgressStatus.STARTED)
        token = _set_environment_lifecycle_progress_reporter(reporter)
        try:
            with pytest.raises(EnvironmentLifecycleDeadlineExceeded):
                await _factory(broker, tmp_path, client, queue_seconds=600).create(_request())
        finally:
            _reset_environment_lifecycle_progress_reporter(token)
        await occupier.aclose()

    asyncio.run(scenario())
    assert published[-1].status is EnvironmentLifecycleProgressStatus.DEADLINE_EXCEEDED
    assert [sandbox.id for sandbox in broker.live()] == ["sbx_0001"]


class _ScriptedQueue:
    """Answers ``queue_position`` and ``get`` from a script, one entry per read."""

    def __init__(self, script: list[Any]) -> None:
        self.script = script
        self.reads = 0

    def _next(self) -> Any:
        self.reads += 1
        return self.script.pop(0) if self.script else None

    async def queue_position(self, sandbox_id: str) -> SandboxQueuedRequest | None:
        answer = self._next()
        if isinstance(answer, BaseException):
            raise answer
        if answer == "unlisted":
            raise SandboxBrokerError("This Cayu Cloud doesn't report sandbox queue positions.")
        if answer is None:
            return None
        position, waiting_for = answer
        return SandboxQueuedRequest(
            id=sandbox_id,
            queued_until=datetime(2026, 10, 28, tzinfo=UTC),
            waiting_for=waiting_for,
            position=position,
        )

    async def get(self, sandbox_id: str) -> SandboxRecord:
        return SandboxRecord.model_validate(
            {
                "id": sandbox_id,
                "image": "coding-python",
                "egress": {"mode": "none"},
                "status": "provisioning",
                "queued_until": "2026-10-28T00:00:00+00:00",
            }
        )


def _report(queue: _ScriptedQueue, reporter: Any, **options: Any) -> None:
    asyncio.run(
        report_sandbox_queue_position(
            queue,  # ty: ignore[invalid-argument-type]
            "sbx_1",
            reporter=reporter,
            poll_interval_s=options.pop("poll_interval_s", 0.001),
            **options,
        )
    )


def test_queue_reports_follow_the_position_and_skip_failed_reads() -> None:
    published: list[EnvironmentLifecycleProgress] = []
    queue = _ScriptedQueue(
        [
            (3, "agent"),
            (3, "agent"),
            SandboxBrokerUnavailableError("down"),
            (2, "organization"),
            (None, "install"),
            "unlisted",
            (None, None),
            None,
        ]
    )
    _report(queue, _reporter(published), heartbeat_s=60)
    # An unchanged position waits for the heartbeat; an unknown one (from Cloud, or from a
    # Cloud without positions) is left out; a place certainly the sandbox's is 0.
    assert [item.queued_count for item in published] == [3, 2, None, 0]
    assert all(item.status is _ADVANCED and item.phase is _PROVISIONING for item in published)
    assert queue.reads == 8


def test_queue_reports_keep_half_the_event_budget_and_need_a_reporter() -> None:
    published: list[EnvironmentLifecycleProgress] = []
    queue = _ScriptedQueue([(1, "agent")] * 100)
    _report(queue, _reporter(published, max_progress_events=32), heartbeat_s=0.001)
    assert len(published) == 16

    # Without a reporter (and none current) nothing is read.
    idle = _ScriptedQueue([(1, "agent")])
    _report(idle, None)
    assert idle.reads == 0
    with pytest.raises(ValueError, match="heartbeat_s"):
        _report(idle, None, heartbeat_s=0)


def test_a_side_failure_wins_over_a_result_in_the_same_loop_turn() -> None:
    async def ready() -> str:
        await asyncio.sleep(0)
        return "ready"

    async def deadline() -> None:
        await asyncio.sleep(0)
        raise TimeoutError("deadline")

    async def scenario() -> None:
        with pytest.raises(TimeoutError, match="deadline"):
            await coding_module._await_beside(ready(), deadline())
        # A side that finishes cleanly leaves the result standing.
        assert await coding_module._await_beside(ready(), asyncio.sleep(0)) == "ready"

    asyncio.run(scenario())
